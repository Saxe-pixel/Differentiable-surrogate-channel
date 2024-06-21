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

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simulation parameters
SEED = 12345
N_SYMBOLS = int(5 * 1e5)
SPS = 8  # Samples-per-symbol (oversampling rate)
BAUD_RATE = 10e6  # Number of symbols transmitted per second
Ts = 1 / BAUD_RATE  # Symbol length
fs = BAUD_RATE * SPS

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Generate data - use Pulse Amplitude Modulation (PAM)
pam_symbols = np.array([-3, -1, 1, 3])
tx_symbols = torch.from_numpy(np.random.choice(pam_symbols, size=(N_SYMBOLS,), replace=True)).double().to(device)

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
    def __init__(self, filter_length, num_filters, initial_non_linear_coefficients=(0,0,0), seed=None):
        super(WHChannelNet, self).__init__()
        self.conv1 = nn.Conv1d(1, num_filters, filter_length, padding=filter_length // 2, bias=False, dtype=torch.double)
        self.conv2 = nn.Conv1d(num_filters, 1, filter_length, padding=filter_length // 2, bias=False, dtype=torch.double)

        self.a0 = nn.Parameter(torch.tensor(initial_non_linear_coefficients[0], dtype=torch.double))
        self.a1 = nn.Parameter(torch.tensor(initial_non_linear_coefficients[1], dtype=torch.double))
        self.a2 = nn.Parameter(torch.tensor(initial_non_linear_coefficients[2], dtype=torch.double))

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.a0 * x + self.a1 * x ** 2 + self.a2 * x ** 3
        x = self.conv2(x)
        return x

class WHChannelNetComplex(nn.Module):
    def __init__(self, filter_length, num_filters, initial_non_linear_coefficients=(0.0, 0.0, 0.0), seed=None):
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

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

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

def train_volterra_model(train_symbols, receiver_rx, h, H, optimizer, channel, num_epochs, batch_size):
    for epoch in range(num_epochs):
        permutation = torch.randperm(train_symbols.size(0))
        for i in range(0, train_symbols.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_tx_symbols = train_symbols[indices].double().to(device)
            optimizer.zero_grad()
            rx, _ = forward_pass(batch_tx_symbols, channel, receiver_rx, h, H, receiver_rx.shape[0] // 2)
            min_length = min(len(rx), len(batch_tx_symbols))
            rx = rx[:min_length]
            batch_tx_symbols = batch_tx_symbols[:min_length]
            loss = F.mse_loss(rx, batch_tx_symbols)
            loss.backward()
            optimizer.step()
        print(f"Volterra - Epoch {epoch + 1}, Loss: {loss.item()}")
    return h, H

def evaluate_volterra_model(tx_symbols_input, receiver_rx, h, H, channel, padding):
    with torch.no_grad():
        tx_symbols_eval = torch.zeros((tx_symbols_input.numel() * SPS,), dtype=torch.double).to(device)
        tx_symbols_eval[0::SPS] = tx_symbols_input.double()
        x = volterra(tx_symbols_eval, h, H)
        y = channel.forward(x)
        rx_eval = F.conv1d(y.view(1, 1, -1), receiver_rx.view(1, 1, -1).flip(dims=[2]), padding=padding).squeeze()
        delay = estimate_delay(rx_eval, SPS)
        rx_eval = rx_eval[delay::SPS]
        rx_eval = rx_eval[:tx_symbols_input.numel()]
        symbols_est = find_closest_symbol(rx_eval, torch.from_numpy(pam_symbols).to(device))
        error = torch.sum(symbols_est != tx_symbols_input[:len(symbols_est)])
        SER = error.float() / len(symbols_est)
        print(f"Volterra - Evaluation SER: {SER.item()}")
        return SER

def train_volterra_receiver_model(train_symbols, tx_pulse, h, H, optimizer, channel, num_epochs, batch_size):
    for epoch in range(num_epochs):
        permutation = torch.randperm(train_symbols.size(0))
        total_loss = 0
        for i in range(0, train_symbols.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_tx_symbols = train_symbols[indices].double().to(device)
            optimizer.zero_grad()

            # Upsample input symbols
            tx_symbols_up = torch.zeros((batch_tx_symbols.numel() * SPS,), dtype=torch.double).to(device)
            tx_symbols_up[0::SPS] = batch_tx_symbols

            # Transmit through the fixed pulse shaping filter
            shaped_pulse = F.conv1d(tx_symbols_up.view(1, 1, -1), tx_pulse.view(1, 1, -1), padding=tx_pulse.shape[0] // 2).squeeze()

            # Pass through the actual WH channel
            y = channel.forward(shaped_pulse)

            # Forward pass through the Volterra receiver
            rx = volterra(y, h, H)

            # Delay estimation and synchronization
            delay = estimate_delay(rx, SPS)
            rx = rx[delay::SPS]
            rx = rx[:batch_tx_symbols.numel()]

            # Loss calculation and backpropagation
            loss = F.mse_loss(rx, batch_tx_symbols)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Volterra Receiver - Epoch {epoch + 1}, Average Loss: {total_loss / (i // batch_size + 1)}")
    return h, H

def evaluate_volterra_receiver_model(tx_symbols_input, tx_pulse, h, H, channel, sps):
    with torch.no_grad():
        # Prepare input signal
        tx_symbols_up = torch.zeros((tx_symbols_input.numel() * sps,), dtype=torch.double).to(device)
        tx_symbols_up[0::sps] = tx_symbols_input.double()

        # Transmit through the fixed pulse shaping filter
        shaped_pulse = F.conv1d(tx_symbols_up.view(1, 1, -1), tx_pulse.view(1, 1, -1), padding=tx_pulse.shape[0] // 2).squeeze()

        # Pass through the actual WH channel
        y = channel.forward(shaped_pulse)

        # Forward pass through the Volterra receiver
        rx_eval = volterra(y, h, H)

        # Delay estimation and synchronization
        delay = estimate_delay(rx_eval, sps)
        symbols_est = rx_eval[delay::sps][:tx_symbols_input.numel()]

        # Symbol decision
        symbols_est = find_closest_symbol(symbols_est, torch.from_numpy(pam_symbols).to(device))

        # Error calculation
        error = torch.sum(symbols_est != tx_symbols_input[:len(symbols_est)])
        SER = error.float() / len(symbols_est)
        print(f"Volterra Receiver - Evaluation SER: {SER.item()}")
        return SER

def train_combined_volterra_model(train_symbols, receiver_rx, h_tx, H_tx, h_rx, H_rx, optimizer, channel, num_epochs, batch_size):
    for epoch in range(num_epochs):
        permutation = torch.randperm(train_symbols.size(0))
        total_loss = 0
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
            rx = volterra(y, h_rx, H_rx)

            # Delay estimation and synchronization
            delay = estimate_delay(rx, SPS)
            rx = rx[delay::SPS]
            rx = rx[:batch_tx_symbols.numel()]

            # Loss calculation and backpropagation
            loss = torch.nn.functional.mse_loss(rx, batch_tx_symbols)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Combined Volterra - Epoch {epoch + 1}, Average Loss: {total_loss / (i // batch_size + 1)}")
    return h_tx, H_tx, h_rx, H_rx

def evaluate_combined_volterra_model(tx_symbols_input, receiver_rx, h_tx, H_tx, h_rx, H_rx, channel, padding):
    with torch.no_grad():
        # Upsample input symbols
        tx_symbols_up = torch.zeros((tx_symbols_input.numel() * SPS,), dtype=torch.double).to(device)
        tx_symbols_up[0::SPS] = tx_symbols_input.double()

        # Apply Volterra series to the transmitter
        shaped_pulse = volterra(tx_symbols_up, h_tx, H_tx)

        # Pass through the actual WH channel
        y = channel.forward(shaped_pulse.squeeze())

        # Apply Volterra series to the receiver
        rx_eval = volterra(y, h_rx, H_rx)

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

def train_model(train_symbols, network, receiver_rx, optimizer, channel, num_epochs, batch_size, sps):
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

def train_receiver_model(train_symbols, tx_pulse, network, optimizer, channel, num_epochs, batch_size, sps):
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

def train_combined_model(train_symbols, network_tx, network_rx, optimizer, channel, num_epochs, batch_size, sps):
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

import random

# Training and evaluation settings
num_epochs = 5
batch_size = 512
num_runs = 5
SNRs = range(0, 21, 2)

# Placeholder for SER results
ser_results = {
    "Volterra Transmitter": [],
    "Volterra Receiver": [],
    "Volterra Combined": [],
    "NN Transmitter": [],
    "NN Transmitter (Complex)": [],
    "NN Receiver": [],
    "NN Receiver (Complex)": [],
    "NN Combined": [],
    "NN Combined (Complex)": []
}

# Function to set seed for reproducibility
def set_seed(run):
    random.seed(run)
    np.random.seed(run)
    torch.manual_seed(run)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run)

# Ranges for different models
h_H_sizes = 44
h_H_sizes_combined = 32
filter_length = 64
num_filters_range = 16
num_filters_range_complex = 5
num_filters_range_combined = 8
num_filters_range_complex_combined = 3

# Train and evaluate models
for snr_db in SNRs:
    print(f"Training and evaluating models for SNR: {snr_db} dB")
    for model_name in ser_results.keys():
        ser_avg = 0.0
        for run in range(num_runs):
            set_seed(run)
            print(f"Run: {run + 1} for model {model_name}")
            if model_name == "Volterra Transmitter":
                h = nn.Parameter(torch.zeros(h_H_sizes, dtype=torch.double, device=device) * 0.01)
                H = nn.Parameter(torch.zeros(h_H_sizes, h_H_sizes, dtype=torch.double, device=device) * 0.01)
                optimizer = optim.Adam([h, H], lr=0.001)
                channel = WienerHammersteinISIChannel(snr_db=snr_db, pulse_energy=pulse_energy, samples_pr_symbol=SPS)
                h, H = train_volterra_model(train_symbols, pulse_rx, h, H, optimizer, channel, num_epochs, batch_size)
                SER = evaluate_volterra_model(test_symbols, pulse_rx, h, H, channel, pulse_rx.shape[0] // 2)
            elif model_name == "Volterra Receiver":
                h = nn.Parameter(torch.zeros(h_H_sizes, dtype=torch.double, device=device) * 0.01)
                H = nn.Parameter(torch.zeros(h_H_sizes, h_H_sizes, dtype=torch.double, device=device) * 0.01)
                optimizer = optim.Adam([h, H], lr=0.001)
                channel = WienerHammersteinISIChannel(snr_db=snr_db, pulse_energy=pulse_energy, samples_pr_symbol=SPS)
                h, H = train_volterra_receiver_model(train_symbols, pulse_tx, h, H, optimizer, channel, num_epochs, batch_size)
                SER = evaluate_volterra_receiver_model(test_symbols, pulse_tx, h, H, channel, SPS)
            elif model_name == "Volterra Combined":
                h_tx = nn.Parameter(torch.zeros(h_H_sizes_combined, dtype=torch.double, device=device) * 0.01)
                H_tx = nn.Parameter(torch.zeros(h_H_sizes_combined, h_H_sizes_combined, dtype=torch.double, device=device) * 0.01)
                h_rx = nn.Parameter(torch.zeros(h_H_sizes_combined, dtype=torch.double, device=device) * 0.01)
                H_rx = nn.Parameter(torch.zeros(h_H_sizes_combined, h_H_sizes_combined, dtype=torch.double, device=device) * 0.01)
                optimizer = optim.Adam([h_tx, H_tx, h_rx, H_rx], lr=0.001)
                channel = WienerHammersteinISIChannel(snr_db=snr_db, pulse_energy=pulse_energy, samples_pr_symbol=SPS)
                h_tx, H_tx, h_rx, H_rx = train_combined_volterra_model(train_symbols, pulse_rx, h_tx, H_tx, h_rx, H_rx, optimizer, channel, num_epochs, batch_size)
                SER = evaluate_combined_volterra_model(test_symbols, pulse_rx, h_tx, H_tx, h_rx, H_rx, channel, pulse_rx.shape[0] // 2)
            elif model_name == "NN Transmitter":
                network = WHChannelNet(FILTER_LENGTH, num_filters_range).to(device)
                optimizer = optim.Adam(network.parameters(), lr=0.001)
                channel = WienerHammersteinISIChannel(snr_db=snr_db, pulse_energy=pulse_energy, samples_pr_symbol=SPS)
                trained_network = train_model(train_symbols, network, pulse_rx, optimizer, channel, num_epochs, batch_size, SPS)
                SER = evaluate_model(test_symbols, trained_network, pulse_rx, channel, SPS)
            elif model_name == "NN Transmitter (Complex)":
                network = WHChannelNetComplex(FILTER_LENGTH, num_filters_range_complex).to(device)
                optimizer = optim.Adam(network.parameters(), lr=0.001)
                channel = WienerHammersteinISIChannel(snr_db=snr_db, pulse_energy=pulse_energy, samples_pr_symbol=SPS)
                trained_network = train_model(train_symbols, network, pulse_rx, optimizer, channel, num_epochs, batch_size, SPS)
                SER = evaluate_model(test_symbols, trained_network, pulse_rx, channel, SPS)
            elif model_name == "NN Receiver":
                network = WHChannelNet(FILTER_LENGTH, num_filters_range).to(device)
                optimizer = optim.Adam(network.parameters(), lr=0.001)
                channel = WienerHammersteinISIChannel(snr_db=snr_db, pulse_energy=pulse_energy, samples_pr_symbol=SPS)
                trained_network = train_receiver_model(train_symbols, pulse_tx, network, optimizer, channel, num_epochs, batch_size, SPS)
                SER = evaluate_receiver_model(test_symbols, pulse_tx, trained_network, channel, SPS)
            elif model_name == "NN Receiver (Complex)":
                network = WHChannelNetComplex(FILTER_LENGTH, num_filters_range_complex).to(device)
                optimizer = optim.Adam(network.parameters(), lr=0.001)
                channel = WienerHammersteinISIChannel(snr_db=snr_db, pulse_energy=pulse_energy, samples_pr_symbol=SPS)
                trained_network = train_receiver_model(train_symbols, pulse_tx, network, optimizer, channel, num_epochs, batch_size, SPS)
                SER = evaluate_receiver_model(test_symbols, pulse_tx, trained_network, channel, SPS)
            elif model_name == "NN Combined":
                network_tx = WHChannelNet(FILTER_LENGTH, num_filters_range_combined).to(device)
                network_rx = WHChannelNet(FILTER_LENGTH, num_filters_range_combined).to(device)
                optimizer = optim.Adam(list(network_tx.parameters()) + list(network_rx.parameters()), lr=0.001)
                channel = WienerHammersteinISIChannel(snr_db=snr_db, pulse_energy=pulse_energy, samples_pr_symbol=SPS)
                trained_network_tx, trained_network_rx = train_combined_model(train_symbols, network_tx, network_rx, optimizer, channel, num_epochs, batch_size, SPS)
                SER = evaluate_combined_model(test_symbols, trained_network_tx, trained_network_rx, channel, SPS)
            elif model_name == "NN Combined (Complex)":
                network_tx = WHChannelNetComplex(FILTER_LENGTH, num_filters_range_complex_combined).to(device)
                network_rx = WHChannelNetComplex(FILTER_LENGTH, num_filters_range_complex_combined).to(device)
                optimizer = optim.Adam(list(network_tx.parameters()) + list(network_rx.parameters()), lr=0.001)
                channel = WienerHammersteinISIChannel(snr_db=snr_db, pulse_energy=pulse_energy, samples_pr_symbol=SPS)
                trained_network_tx, trained_network_rx = train_combined_model(train_symbols, network_tx, network_rx, optimizer, channel, num_epochs, batch_size, SPS)
                SER = evaluate_combined_model(test_symbols, trained_network_tx, trained_network_rx, channel, SPS)
            ser_avg += SER.item()
        ser_avg /= num_runs
        ser_results[model_name].append(ser_avg)

import matplotlib.pyplot as plt

def plot_results(ser_results, label, marker, linestyle, color):
    plt.plot(SNRs, ser_results, label=label, marker=marker, linestyle=linestyle, color=color)

# Assuming ser_results dictionary contains your results for each model
# Example of how to call plot_results for each model with the specified color codes

# Volterra
plot_results(ser_results["Volterra Transmitter"], "Volterra Transmitter", 'o', '-', '#1f77b4')  # muted blue
plot_results(ser_results["Volterra Receiver"], "Volterra Receiver", 'd', '--', '#1f99d4')  # slightly lighter blue
plot_results(ser_results["Volterra Combined"], "Volterra Combined", 's', '-.', '#1f55b4')  # slightly darker blue

# Transmitter
plot_results(ser_results["NN Transmitter"], "NN Transmitter", 'o', '-', '#F06000')  # muted orange
plot_results(ser_results["NN Transmitter (Complex)"], "NN Transmitter (Complex)", 'o', '-', '#FF8F00')  # light orange

# Receiver
plot_results(ser_results["NN Receiver"], "NN Receiver", 'd', '--', '#F29F05')  # muted red
plot_results(ser_results["NN Receiver (Complex)"], "NN Receiver (Complex)", 'd', '--', '#FDD430')  # slightly darker red

# Combined
plot_results(ser_results["NN Combined"], "NN Combined", 's', '-.', 'green')  # muted green
plot_results(ser_results["NN Combined (Complex)"], "NN Combined (Complex)", 's', '-.', '#B4CF66')  # slightly lighter green

# Final plot adjustments
plt.xlabel("SNR (dB)")
plt.ylabel("SER")
plt.yscale("log")
plt.title("SER vs. SNR for Different Models")
plt.legend(ncol=2)
plt.grid()
plt.show()