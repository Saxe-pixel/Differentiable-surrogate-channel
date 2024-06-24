import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from commpy.filters import rrcosfilter
from lib.utility import estimate_delay, find_closest_symbol
from lib.channels import WienerHammersteinISIChannel
import torch.optim.lr_scheduler as lr_scheduler
from skopt import gp_minimize
from skopt.space import Integer, Real

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simulation parameters
SEED = 12345
N_SYMBOLS = int(2 * 1e5)
SPS = 8  # Samples-per-symbol (oversampling rate)
BAUD_RATE = 10e6  # Number of symbols transmitted per second
Ts = 1 / BAUD_RATE  # Symbol length
fs = BAUD_RATE * SPS
random_obj = np.random.default_rng(SEED)

# Generate data - use Pulse Amplitude Modulation (PAM)
pam_symbols = np.array([-3, -1, 1, 3])
tx_symbols = torch.from_numpy(random_obj.choice(pam_symbols, size=(N_SYMBOLS,), replace=True)).double().to(device)

# Split the data into training and testing
train_size = int(0.7 * N_SYMBOLS)
test_size = N_SYMBOLS - train_size
train_symbols = tx_symbols[:train_size]
test_symbols = tx_symbols[train_size:]

# Construct pulse shape for both transmitter and receiver
FILTER_LENGTH = 64
t, g = rrcosfilter(FILTER_LENGTH, 0.5, Ts, fs)
g /= np.linalg.norm(g)  # Normalize pulse to have unit energy
pulse_rx = torch.from_numpy(g).double().to(device)  # Receiver pulse (fixed)
pulse_tx = torch.from_numpy(g).double().to(device)  # Transmitter pulse (fixed)
pulse_energy = np.max(np.convolve(g, g[::-1]))

def volterra(x, h, H):
    T = len(x)
    y_t = torch.zeros(T, dtype=torch.double, device=device)

    # Linear terms
    conv_x_h = F.conv1d(x.view(1, 1, -1), h.view(1, 1, -1), padding=h.size(0) - 1).squeeze()
    y_t += conv_x_h[:T]

    # Second-order non-linear terms
    valid_length = T - H.shape[0] + 1
    for i in range(H.shape[0]):
        y_t[i:valid_length + i] += torch.sum(H[i, :].view(-1, 1) * x[:valid_length] * x[i:i + valid_length], dim=0)

    return y_t

def forward_pass(tx_symbols_input, channel, receiver_rx, h, H, padding):
    # Upsample
    tx_symbols_up = torch.zeros((tx_symbols_input.numel() * SPS,), dtype=torch.double, device=device)
    tx_symbols_up[0::SPS] = tx_symbols_input.double()

    # Apply optimized Volterra series
    x = volterra(tx_symbols_up, h, H)

    # Simulate the channel
    y = channel.forward(x)

    # Apply receiver with consistent padding
    rx = F.conv1d(y.view(1, 1, -1), receiver_rx.view(1, 1, -1).flip(dims=[2]), padding=padding).squeeze()
    delay = estimate_delay(rx, SPS)
    rx = rx[delay::SPS]
    rx = rx[:tx_symbols_input.numel()]
    return rx, tx_symbols_up

class WHChannelNet(nn.Module):
    def __init__(self, filter_length, num_filters, initial_non_linear_coefficients=(0,0,0)):
        super(WHChannelNet, self).__init__()
        self.conv1 = nn.Conv1d(1, num_filters, filter_length, padding=filter_length // 2, bias=False, dtype=torch.double)
        self.conv2 = nn.Conv1d(num_filters, 1, filter_length, padding=filter_length // 2, bias=False, dtype=torch.double)

        self.a0 = nn.Parameter(torch.tensor(initial_non_linear_coefficients[0], dtype=torch.double))
        self.a1 = nn.Parameter(torch.tensor(initial_non_linear_coefficients[1], dtype=torch.double))
        self.a2 = nn.Parameter(torch.tensor(initial_non_linear_coefficients[2], dtype=torch.double))

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.a0 * x + self.a1 * x ** 2 + self.a2 * x ** 3
        x = self.conv2(x)
        return x

class WHChannelNetComplex(nn.Module):
    def __init__(self, filter_length, num_filters, initial_non_linear_coefficients=(0.0, 0.0, 0.0)):
        super(WHChannelNetComplex, self).__init__()
        self.conv1 = nn.Conv1d(1, num_filters, filter_length, padding=filter_length // 2, bias=False, dtype=torch.double)
        self.conv2 = nn.Conv1d(num_filters, num_filters, filter_length, padding=filter_length // 2, bias=False, dtype=torch.double)
        self.conv3 = nn.Conv1d(num_filters, 1, filter_length, padding=filter_length // 2, bias=False, dtype=torch.double)

        # Adding batch normalization layers
        self.bn1 = nn.BatchNorm1d(num_filters, dtype=torch.double)
        self.bn2 = nn.BatchNorm1d(num_filters, dtype=torch.double)
        
        # Adding residual connections
        self.residual = nn.Conv1d(1, num_filters, 1, bias=False, dtype=torch.double)
        
        # Initialize the non-linear coefficients as learnable parameters
        self.a0 = nn.Parameter(torch.tensor(initial_non_linear_coefficients[0], dtype=torch.double))
        self.a1 = nn.Parameter(torch.tensor(initial_non_linear_coefficients[1], dtype=torch.double))
        self.a2 = nn.Parameter(torch.tensor(initial_non_linear_coefficients[2], dtype=torch.double))

        # Initialize the weights of the convolutional layers
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.residual.weight)

    def forward(self, x):
        res = self.residual(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = x[:, :, :res.shape[2]]  # Ensure the size matches for residual connection
        x = x + res  # Residual connection
        x = self.a0 * x + self.a1 * x ** 2 + self.a2 * x ** 3
        x = self.conv3(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Training and evaluation functions (same as in the initial implementation)

def train_volterra_transmitter_model(train_symbols, receiver_rx, h, H, optimizer, scheduler, channel, num_epochs, batch_size):
    for epoch in range(num_epochs):
        permutation = torch.randperm(train_size, device=device)
        for i in range(0, train_size, batch_size):
            indices = permutation[i:i+batch_size]
            batch_tx_symbols = train_symbols[indices].double()

            optimizer.zero_grad()
            rx, _ = forward_pass(batch_tx_symbols, channel, receiver_rx, h, H, receiver_rx.shape[0] // 2)

            # Compute loss
            min_length = min(len(rx), len(batch_tx_symbols))
            rx = rx[:min_length]
            batch_tx_symbols = batch_tx_symbols[:min_length]

            loss = F.mse_loss(rx, batch_tx_symbols)  # Mean squared error loss
            loss.backward()
            optimizer.step()
        scheduler.step()  # Update the learning rate
        print(f"Volterra Transmitter - Epoch {epoch+1}, Loss: {loss.item()}")
    return h, H

def evaluate_volterra_transmitter_model(tx_symbols_input, receiver_rx, h, H, channel, padding):
    with torch.no_grad():
        tx_symbols_eval = torch.zeros((tx_symbols_input.numel() * SPS,), dtype=torch.double, device=device)
        tx_symbols_eval[0::SPS] = tx_symbols_input.double()

        # Apply Volterra series on the transmitter side
        x = volterra(tx_symbols_eval, h, H)

        # Simulate the channel
        y = channel.forward(x)

        # Apply receiver with consistent padding
        rx_eval = F.conv1d(y.view(1, 1, -1), receiver_rx.view(1, 1, -1).flip(dims=[2]), padding=padding).squeeze()

        delay = estimate_delay(rx_eval, SPS)
        rx_eval = rx_eval[delay::SPS]
        rx_eval = rx_eval[:tx_symbols_input.numel()]

        # Symbol estimation
        symbols_est = find_closest_symbol(rx_eval, torch.from_numpy(pam_symbols).to(device))

        # Calculate errors and SER
        error = torch.sum(symbols_est != tx_symbols_input[:len(symbols_est)])
        SER = error.float() / len(symbols_est)
        print(f"Volterra Transmitter - Evaluation SER: {SER.item()}")
        return SER

def train_volterra_receiver_model(train_symbols, receiver_rx, h_rx, H_rx, optimizer, scheduler, channel, num_epochs, batch_size, pulse_tx):
    if receiver_rx.ndim == 1:
        receiver_rx_flipped = receiver_rx.flip(dims=[0])  # Pre-flip the receiver filter
    else:
        receiver_rx_flipped = receiver_rx.flip(dims=[-1])  # Pre-flip the receiver filter for higher dimensions

    pulse_tx_padded = pulse_tx.shape[0] // 2

    for epoch in range(num_epochs):
        permutation = torch.randperm(train_symbols.size(0), device=device)
        total_loss = 0.0
        num_batches = 0

        for i in range(0, train_symbols.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_tx_symbols = train_symbols[indices].double().to(device)
            optimizer.zero_grad()

            # Upsample input symbols
            tx_symbols_up = torch.zeros((batch_tx_symbols.numel() * SPS,), dtype=torch.double).to(device)
            tx_symbols_up[0::SPS] = batch_tx_symbols

            # Apply RRC filter to the transmitter
            shaped_pulse = F.conv1d(tx_symbols_up.view(1, 1, -1), pulse_tx.view(1, 1, -1).flip(dims=[2]), padding=pulse_tx_padded).squeeze()

            # Pass through the actual WH channel
            y = channel.forward(shaped_pulse)

            # Apply Volterra series to the receiver
            rx_volterra = volterra(y, h_rx, H_rx)

            # Apply receiver filter with consistent padding
            rx = F.conv1d(rx_volterra.view(1, 1, -1), receiver_rx_flipped.view(1, 1, -1), padding=receiver_rx.shape[0] // 2).squeeze()

            # Delay estimation and synchronization
            delay = estimate_delay(rx, SPS)
            rx = rx[delay::SPS][:batch_tx_symbols.numel()]

            # Loss calculation and backpropagation
            loss = F.mse_loss(rx, batch_tx_symbols)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()  # Update the learning rate
        print(f"Volterra Receiver - Epoch {epoch + 1}, Average Loss: {total_loss / num_batches}")

    return h_rx, H_rx

def evaluate_volterra_receiver_model(tx_symbols_input, receiver_rx, h_rx, H_rx, channel, pulse_tx, pam_symbols):
    with torch.no_grad():
        # Upsample input symbols
        tx_symbols_up = torch.zeros((tx_symbols_input.numel() * SPS,), dtype=torch.double).to(device)
        tx_symbols_up[0::SPS] = tx_symbols_input

        # Apply RRC filter to the transmitter
        shaped_pulse = F.conv1d(tx_symbols_up.view(1, 1, -1), pulse_tx.view(1, 1, -1).flip(dims=[2]), padding=pulse_tx.shape[0] // 2).squeeze()

        # Pass through the actual WH channel
        y = channel.forward(shaped_pulse)

        # Apply Volterra series to the receiver
        rx_volterra = volterra(y, h_rx, H_rx)

        # Apply receiver filter with consistent padding
        if receiver_rx.ndim == 1:
            receiver_rx_flipped = receiver_rx.flip(dims=[0])  # Pre-flip the receiver filter
        else:
            receiver_rx_flipped = receiver_rx.flip(dims=[-1])  # Pre-flip the receiver filter for higher dimensions

        rx_eval = F.conv1d(rx_volterra.view(1, 1, -1), receiver_rx_flipped.view(1, 1, -1), padding=receiver_rx.shape[0] // 2).squeeze()

        # Delay estimation and synchronization
        delay = estimate_delay(rx_eval, SPS)
        rx_eval = rx_eval[delay::SPS][:tx_symbols_input.numel()]

        # Symbol decision
        symbols_est = find_closest_symbol(rx_eval, torch.from_numpy(pam_symbols).to(device))

        # Error calculation
        error = torch.sum(symbols_est != tx_symbols_input[:len(symbols_est)])
        SER = error.float() / len(symbols_est)
        print(f"Volterra Receiver - Evaluation SER: {SER.item()}")
        return SER

def train_combined_volterra_model(train_symbols, receiver_rx, h_tx, H_tx, h_rx, H_rx, optimizer, scheduler, channel, num_epochs, batch_size):
    for epoch in range(num_epochs):
        permutation = torch.randperm(train_symbols.size(0))
        total_loss = 0
        num_batches = 0
        for i in range(0, train_symbols.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_tx_symbols = train_symbols[indices].double().to(device)
            optimizer.zero_grad()

            # Upsample input symbols
            tx_symbols_up = torch.zeros((batch_tx_symbols.numel() * SPS,), dtype=torch.double).to(device)
            tx_symbols_up[0::SPS] = batch_tx_symbols

            # Apply Volterra series to the transmitter
            shaped_pulse = volterra(tx_symbols_up, h_tx, H_tx)

            # Pass through the actual WH channel
            y = channel.forward(shaped_pulse.squeeze())

            # Apply Volterra series to the receiver
            rx_volterra = volterra(y, h_rx, H_rx)

            # Apply receiver filter with consistent padding
            rx = F.conv1d(rx_volterra.view(1, 1, -1), receiver_rx.view(1, 1, -1).flip(dims=[2]), padding=receiver_rx.shape[0] // 2).squeeze()

            # Delay estimation and synchronization
            delay = estimate_delay(rx, SPS)
            rx = rx[delay::SPS]
            rx = rx[:batch_tx_symbols.numel()]

            # Loss calculation and backpropagation
            loss = F.mse_loss(rx, batch_tx_symbols)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()  # Update the learning rate
        print(f"Combined Volterra - Epoch {epoch + 1}, Average Loss: {total_loss / num_batches}")
    return h_tx, H_tx, h_rx, H_rx

def evaluate_combined_volterra_model(tx_symbols_input, receiver_rx, h_tx, H_tx, h_rx, H_rx, channel):
    with torch.no_grad():
        # Upsample input symbols
        tx_symbols_up = torch.zeros((tx_symbols_input.numel() * SPS,), dtype=torch.double).to(device)
        tx_symbols_up[0::SPS] = tx_symbols_input.double()

        # Apply Volterra series to the transmitter
        shaped_pulse = volterra(tx_symbols_up, h_tx, H_tx)

        # Pass through the actual WH channel
        y = channel.forward(shaped_pulse.squeeze())

        # Apply Volterra series to the receiver
        rx_volterra = volterra(y, h_rx, H_rx)

        # Apply receiver filter with consistent padding
        rx_eval = F.conv1d(rx_volterra.view(1, 1, -1), receiver_rx.view(1, 1, -1).flip(dims=[2]), padding=receiver_rx.shape[0] // 2).squeeze()

        # Delay estimation and synchronization
        delay = estimate_delay(rx_eval, SPS)
        rx_eval = rx_eval[delay::SPS]
        rx_eval = rx_eval[:tx_symbols_input.numel()]

        # Symbol decision
        symbols_est = find_closest_symbol(rx_eval, torch.from_numpy(pam_symbols).to(device))

        # Error calculation
        error = torch.sum(symbols_est != tx_symbols_input[:len(symbols_est)])
        SER = error.float() / len(symbols_est)
        print(f"Combined Volterra - Evaluation SER: {SER.item()}")
        return SER

def train_model(train_symbols, network, receiver_rx, optimizer, scheduler, channel, num_epochs, batch_size, sps):
    for epoch in range(num_epochs):
        permutation = torch.randperm(train_symbols.size(0))
        total_loss = 0
        for i in range(0, train_symbols.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_tx_symbols = train_symbols[indices].double().to(device)
            optimizer.zero_grad()
            tx_symbols_up = torch.zeros((batch_tx_symbols.numel() * sps,), dtype=torch.double).to(device)
            tx_symbols_up[0::sps] = batch_tx_symbols
            shaped_pulse = network(tx_symbols_up.view(1, 1, -1))
            y = channel.forward(shaped_pulse.squeeze())
            rx = torch.nn.functional.conv1d(y.view(1, 1, -1), receiver_rx.view(1, 1, -1).flip(dims=[2]), padding=receiver_rx.shape[0] // 2).squeeze()
            delay = estimate_delay(rx, sps)
            rx = rx[delay::sps]
            rx = rx[:batch_tx_symbols.numel()]
            loss = torch.nn.functional.mse_loss(rx, batch_tx_symbols)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()  # Update the learning rate
        print(f"Transmitter - Epoch {epoch + 1}, Average Loss: {total_loss / (i // batch_size + 1)}")
    return network

def evaluate_model(tx_symbols_input, network, receiver_rx, channel, sps):
    with torch.no_grad():
        tx_symbols_up = torch.zeros((tx_symbols_input.numel() * sps,), dtype=torch.double).to(device)
        tx_symbols_up[0::sps] = tx_symbols_input.double()
        shaped_pulse = network(tx_symbols_up.view(1, 1, -1))
        y = channel.forward(shaped_pulse.squeeze())
        rx_eval = torch.nn.functional.conv1d(y.view(1, 1, -1), receiver_rx.view(1, 1, -1).flip(dims=[2]), padding=receiver_rx.shape[0] // 2).squeeze()
        delay = estimate_delay(rx_eval, sps)
        symbols_est = rx_eval[delay::sps][:tx_symbols_input.numel()]
        symbols_est = find_closest_symbol(symbols_est, torch.from_numpy(pam_symbols))
        error = torch.sum(torch.logical_not(torch.eq(symbols_est, tx_symbols_input)))
        SER = error.float() / tx_symbols_input.numel()
        print(f"Transmitter - Evaluation SER: {SER.item()}")
        return SER

def train_receiver_model(train_symbols, tx_pulse, network, optimizer, scheduler, channel, num_epochs, batch_size, sps):
    for epoch in range(num_epochs):
        permutation = torch.randperm(train_symbols.size(0))
        total_loss = 0
        for i in range(0, train_symbols.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_tx_symbols = train_symbols[indices].double().to(device)
            optimizer.zero_grad()

            # Upsample input symbols
            tx_symbols_up = torch.zeros((batch_tx_symbols.numel() * sps,), dtype=torch.double).to(device)
            tx_symbols_up[0::sps] = batch_tx_symbols

            # Transmit through the fixed pulse shaping filter
            shaped_pulse = torch.nn.functional.conv1d(tx_symbols_up.view(1, 1, -1), tx_pulse.view(1, 1, -1), padding=tx_pulse.shape[0] // 2)

            # Pass through the actual WH channel
            y = channel.forward(shaped_pulse.squeeze())

            # Forward pass through the receiver network
            rx = network(y.view(1, 1, -1)).squeeze()

            # Delay estimation and synchronization
            delay = estimate_delay(rx, sps)
            rx = rx[delay::sps]
            rx = rx[:batch_tx_symbols.numel()]

            # Loss calculation and backpropagation
            loss = torch.nn.functional.mse_loss(rx, batch_tx_symbols)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        scheduler.step()  # Update the learning rate
        print(f"Receiver - Epoch {epoch + 1}, Average Loss: {total_loss / (i // batch_size + 1)}")
    return network

def evaluate_receiver_model(tx_symbols_input, tx_pulse, network, channel, sps):
    with torch.no_grad():
        # Prepare input signal
        tx_symbols_up = torch.zeros((tx_symbols_input.numel() * sps,), dtype=torch.double).to(device)
        tx_symbols_up[0::sps] = tx_symbols_input.double()

        # Transmit through the fixed pulse shaping filter
        shaped_pulse = torch.nn.functional.conv1d(tx_symbols_up.view(1, 1, -1), tx_pulse.view(1, 1, -1), padding=tx_pulse.shape[0] // 2)

        # Pass through the actual WH channel
        y = channel.forward(shaped_pulse.squeeze())

        # Forward pass through the receiver network
        rx_eval = network(y.view(1, 1, -1)).squeeze()

        # Delay estimation and synchronization
        delay = estimate_delay(rx_eval, sps)
        symbols_est = rx_eval[delay::sps][:tx_symbols_input.numel()]

        # Symbol decision
        symbols_est = find_closest_symbol(symbols_est, torch.from_numpy(pam_symbols))

        # Error calculation
        error = torch.sum(torch.logical_not(torch.eq(symbols_est, tx_symbols_input)))
        SER = error.float() / tx_symbols_input.numel()
        print(f"Receiver - Evaluation SER: {SER.item()}")
        return SER

def train_combined_model(train_symbols, network_tx, network_rx, optimizer, scheduler, channel, num_epochs, batch_size, sps):
    for epoch in range(num_epochs):
        permutation = torch.randperm(train_symbols.size(0))
        total_loss = 0
        for i in range(0, train_symbols.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_tx_symbols = train_symbols[indices].double().to(device)
            optimizer.zero_grad()

            # Upsample input symbols
            tx_symbols_up = torch.zeros((batch_tx_symbols.numel() * sps,), dtype=torch.double).to(device)
            tx_symbols_up[0::sps] = batch_tx_symbols

            # Forward pass through the transmitter network
            shaped_pulse = network_tx(tx_symbols_up.view(1, 1, -1))

            # Pass through the actual WH channel
            y = channel.forward(shaped_pulse.squeeze())

            # Forward pass through the receiver network
            rx = network_rx(y.view(1, 1, -1)).squeeze()

            # Delay estimation and synchronization
            delay = estimate_delay(rx, sps)
            rx = rx[delay::sps]
            rx = rx[:batch_tx_symbols.numel()]

            # Loss calculation and backpropagation
            loss = torch.nn.functional.mse_loss(rx, batch_tx_symbols)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        scheduler.step()  # Update the learning rate
        print(f"Combined - Epoch {epoch + 1}, Average Loss: {total_loss / (i // batch_size + 1)}")
    return network_tx, network_rx

def evaluate_combined_model(tx_symbols_input, network_tx, network_rx, channel, sps):
    with torch.no_grad():
        # Prepare input signal
        tx_symbols_up = torch.zeros((tx_symbols_input.numel() * sps,), dtype=torch.double).to(device)
        tx_symbols_up[0::sps] = tx_symbols_input.double()

        # Forward pass through the transmitter network
        shaped_pulse = network_tx(tx_symbols_up.view(1, 1, -1))

        # Pass through the actual WH channel
        y = channel.forward(shaped_pulse.squeeze())

        # Forward pass through the receiver network
        rx_eval = network_rx(y.view(1, 1, -1)).squeeze()

        # Delay estimation and synchronization
        delay = estimate_delay(rx_eval, sps)
        symbols_est = rx_eval[delay::sps][:tx_symbols_input.numel()]

        # Symbol decision
        symbols_est = find_closest_symbol(symbols_est, torch.from_numpy(pam_symbols))

        # Error calculation
        error = torch.sum(torch.logical_not(torch.eq(symbols_est, tx_symbols_input)))
        SER = error.float() / tx_symbols_input.numel()
        print(f"Combined - Evaluation SER: {SER.item()}")
        return SER

# Bayesian Optimization for different models

def bayesian_optimize_volterra_transmitter(train_symbols, test_symbols, receiver_rx, sps, pulse_energy):
    def objective(params):
        size = params[0]
        h = nn.Parameter(torch.zeros(size, dtype=torch.double, device=device) * 0.01)
        H = nn.Parameter(torch.zeros(size, size, dtype=torch.double, device=device) * 0.01)
        optimizer = torch.optim.Adam([h, H], lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        channel = WienerHammersteinISIChannel(snr_db=5, pulse_energy=pulse_energy, samples_pr_symbol=sps)
        h, H = train_volterra_transmitter_model(train_symbols, receiver_rx, h, H, optimizer, scheduler, channel, num_epochs=5, batch_size=512)
        SER = evaluate_volterra_transmitter_model(test_symbols, receiver_rx, h, H, channel, receiver_rx.shape[0] // 2)
        return SER.item()

    res = gp_minimize(objective, [Integer(4, 64)], n_calls=15, random_state=SEED)
    return res

def bayesian_optimize_volterra_receiver(train_symbols, test_symbols, receiver_rx, sps, pulse_energy, pulse_tx):
    def objective(params):
        size = params[0]
        h_rx = nn.Parameter(torch.zeros(size, dtype=torch.double, device=device) * 0.01)
        H_rx = nn.Parameter(torch.zeros(size, size, dtype=torch.double, device=device) * 0.01)
        optimizer = torch.optim.Adam([h_rx, H_rx], lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        channel = WienerHammersteinISIChannel(snr_db=5, pulse_energy=pulse_energy, samples_pr_symbol=sps)
        h_rx, H_rx = train_volterra_receiver_model(train_symbols, receiver_rx, h_rx, H_rx, optimizer, scheduler, channel, num_epochs=5, batch_size=512, pulse_tx=pulse_tx)
        SER = evaluate_volterra_receiver_model(test_symbols, receiver_rx, h_rx, H_rx, channel, pulse_tx, pam_symbols)
        return SER.item()

    res = gp_minimize(objective, [Integer(4, 64)], n_calls=15, random_state=SEED)
    return res

def bayesian_optimize_combined_volterra(train_symbols, test_symbols, receiver_rx, sps, pulse_energy):
    def objective(params):
        size = params[0]
        h_tx = nn.Parameter(torch.zeros(size, dtype=torch.double, device=device) * 0.01)
        H_tx = nn.Parameter(torch.zeros(size, size, dtype=torch.double, device=device) * 0.01)
        h_rx = nn.Parameter(torch.zeros(size, dtype=torch.double, device=device) * 0.01)
        H_rx = nn.Parameter(torch.zeros(size, size, dtype=torch.double, device=device) * 0.01)
        optimizer = torch.optim.Adam([h_tx, H_tx, h_rx, H_rx], lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        channel = WienerHammersteinISIChannel(snr_db=5, pulse_energy=pulse_energy, samples_pr_symbol=sps)
        h_tx, H_tx, h_rx, H_rx = train_combined_volterra_model(train_symbols, receiver_rx, h_tx, H_tx, h_rx, H_rx, optimizer, scheduler, channel, num_epochs=5, batch_size=512)
        SER = evaluate_combined_volterra_model(test_symbols, receiver_rx, h_tx, H_tx, h_rx, H_rx, channel)
        return SER.item()

    res = gp_minimize(objective, [Integer(4, 64)], n_calls=15, random_state=SEED)
    return res

def bayesian_optimize_whchannelnet(train_symbols, test_symbols, receiver_rx, sps, pulse_energy):
    def objective(params):
        filter_length = params[0]
        num_filters = params[1]
        network = WHChannelNet(filter_length, num_filters).to(device)
        optimizer = optim.Adam(network.parameters(), lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        channel = WienerHammersteinISIChannel(snr_db=5, pulse_energy=pulse_energy, samples_pr_symbol=sps)
        trained_network = train_model(train_symbols, network, receiver_rx, optimizer, scheduler, channel, num_epochs=5, batch_size=512, sps=sps)
        SER = evaluate_model(test_symbols, trained_network, receiver_rx, channel, sps)
        return SER.item()

    res = gp_minimize(objective, [Integer(8, 128), Integer(1, 64)], n_calls=15, random_state=SEED)
    return res

def bayesian_optimize_whchannelnet_complex(train_symbols, test_symbols, receiver_rx, sps, pulse_energy):
    def objective(params):
        filter_length = params[0]
        num_filters = params[1]
        network = WHChannelNetComplex(filter_length, num_filters).to(device)
        optimizer = optim.Adam(network.parameters(), lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        channel = WienerHammersteinISIChannel(snr_db=5, pulse_energy=pulse_energy, samples_pr_symbol=sps)
        trained_network = train_model(train_symbols, network, receiver_rx, optimizer, scheduler, channel, num_epochs=5, batch_size=512, sps=sps)
        SER = evaluate_model(test_symbols, trained_network, receiver_rx, channel, sps)
        return SER.item()

    res = gp_minimize(objective, [Integer(8, 128), Integer(1, 64)], n_calls=15, random_state=SEED)
    return res

def bayesian_optimize_whchannelnet_receiver(train_symbols, test_symbols, receiver_rx, sps, pulse_energy, pulse_tx):
    def objective(params):
        filter_length = params[0]
        num_filters = params[1]
        network = WHChannelNet(filter_length, num_filters).to(device)
        optimizer = optim.Adam(network.parameters(), lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        channel = WienerHammersteinISIChannel(snr_db=5, pulse_energy=pulse_energy, samples_pr_symbol=sps)
        trained_network = train_receiver_model(train_symbols, pulse_tx, network, optimizer, scheduler, channel, num_epochs=5, batch_size=512, sps=sps)
        SER = evaluate_receiver_model(test_symbols, pulse_tx, trained_network, channel, sps)
        return SER.item()

    res = gp_minimize(objective, [Integer(8, 128), Integer(1, 64)], n_calls=15, random_state=SEED)
    return res

def bayesian_optimize_whchannelnet_complex_receiver(train_symbols, test_symbols, receiver_rx, sps, pulse_energy, pulse_tx):
    def objective(params):
        filter_length = params[0]
        num_filters = params[1]
        network = WHChannelNetComplex(filter_length, num_filters).to(device)
        optimizer = optim.Adam(network.parameters(), lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        channel = WienerHammersteinISIChannel(snr_db=5, pulse_energy=pulse_energy, samples_pr_symbol=sps)
        trained_network = train_receiver_model(train_symbols, pulse_tx, network, optimizer, scheduler, channel, num_epochs=5, batch_size=512, sps=sps)
        SER = evaluate_receiver_model(test_symbols, pulse_tx, trained_network, channel, sps)
        return SER.item()

    res = gp_minimize(objective, [Integer(8, 128), Integer(1, 64)], n_calls=15, random_state=SEED)
    return res

def bayesian_optimize_combined_whchannelnet(train_symbols, test_symbols, receiver_rx, sps, pulse_energy):
    def objective(params):
        filter_length = params[0]
        num_filters = params[1]
        network_tx = WHChannelNet(filter_length, num_filters).to(device)
        network_rx = WHChannelNet(filter_length, num_filters).to(device)
        optimizer = optim.Adam(list(network_tx.parameters()) + list(network_rx.parameters()), lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        channel = WienerHammersteinISIChannel(snr_db=5, pulse_energy=pulse_energy, samples_pr_symbol=sps)
        trained_network_tx, trained_network_rx = train_combined_model(train_symbols, network_tx, network_rx, optimizer, scheduler, channel, num_epochs=5, batch_size=512, sps=sps)
        SER = evaluate_combined_model(test_symbols, trained_network_tx, trained_network_rx, channel, sps)
        return SER.item()

    res = gp_minimize(objective, [Integer(8, 128), Integer(1, 64)], n_calls=15, random_state=SEED)
    return res

def bayesian_optimize_combined_whchannelnet_complex(train_symbols, test_symbols, receiver_rx, sps, pulse_energy):
    def objective(params):
        filter_length = params[0]
        num_filters = params[1]
        network_tx = WHChannelNetComplex(filter_length, num_filters).to(device)
        network_rx = WHChannelNetComplex(filter_length, num_filters).to(device)
        optimizer = optim.Adam(list(network_tx.parameters()) + list(network_rx.parameters()), lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        channel = WienerHammersteinISIChannel(snr_db=5, pulse_energy=pulse_energy, samples_pr_symbol=sps)
        trained_network_tx, trained_network_rx = train_combined_model(train_symbols, network_tx, network_rx, optimizer, scheduler, channel, num_epochs=5, batch_size=512, sps=sps)
        SER = evaluate_combined_model(test_symbols, trained_network_tx, trained_network_rx, channel, sps)
        return SER.item()

    res = gp_minimize(objective, [Integer(8, 128), Integer(1, 64)], n_calls=15, random_state=SEED)
    return res

# Run Bayesian Optimization for different models

volterra_transmitter_res = bayesian_optimize_volterra_transmitter(train_symbols, test_symbols, pulse_rx, SPS, pulse_energy)
print("Best parameters for Volterra Transmitter: ", volterra_transmitter_res.x)
print("Best SER for Volterra Transmitter: ", volterra_transmitter_res.fun)

volterra_receiver_res = bayesian_optimize_volterra_receiver(train_symbols, test_symbols, pulse_rx, SPS, pulse_energy, pulse_tx)
print("Best parameters for Volterra Receiver: ", volterra_receiver_res.x)
print("Best SER for Volterra Receiver: ", volterra_receiver_res.fun)

combined_volterra_res = bayesian_optimize_combined_volterra(train_symbols, test_symbols, pulse_rx, SPS, pulse_energy)
print("Best parameters for Combined Volterra: ", combined_volterra_res.x)
print("Best SER for Combined Volterra: ", combined_volterra_res.fun)

whchannelnet_res = bayesian_optimize_whchannelnet(train_symbols, test_symbols, pulse_rx, SPS, pulse_energy)
print("Best parameters for WHChannelNet: ", whchannelnet_res.x)
print("Best SER for WHChannelNet: ", whchannelnet_res.fun)

whchannelnet_complex_res = bayesian_optimize_whchannelnet_complex(train_symbols, test_symbols, pulse_rx, SPS, pulse_energy)
print("Best parameters for WHChannelNetComplex: ", whchannelnet_complex_res.x)
print("Best SER for WHChannelNetComplex: ", whchannelnet_complex_res.fun)

whchannelnet_receiver_res = bayesian_optimize_whchannelnet_receiver(train_symbols, test_symbols, pulse_rx, SPS, pulse_energy, pulse_tx)
print("Best parameters for WHChannelNet Receiver: ", whchannelnet_receiver_res.x)
print("Best SER for WHChannelNet Receiver: ", whchannelnet_receiver_res.fun)

whchannelnet_complex_receiver_res = bayesian_optimize_whchannelnet_complex_receiver(train_symbols, test_symbols, pulse_rx, SPS, pulse_energy, pulse_tx)
print("Best parameters for WHChannelNetComplex Receiver: ", whchannelnet_complex_receiver_res.x)
print("Best SER for WHChannelNetComplex Receiver: ", whchannelnet_complex_receiver_res.fun)

combined_whchannelnet_res = bayesian_optimize_combined_whchannelnet(train_symbols, test_symbols, pulse_rx, SPS, pulse_energy)
print("Best parameters for Combined WHChannelNet: ", combined_whchannelnet_res.x)
print("Best SER for Combined WHChannelNet: ", combined_whchannelnet_res.fun)

combined_whchannelnet_complex_res = bayesian_optimize_combined_whchannelnet_complex(train_symbols, test_symbols, pulse_rx, SPS, pulse_energy)
print("Best parameters for Combined WHChannelNetComplex: ", combined_whchannelnet_complex_res.x)
print("Best SER for Combined WHChannelNetComplex: ", combined_whchannelnet_complex_res.fun)
