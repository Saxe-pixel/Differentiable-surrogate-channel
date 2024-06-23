import numpy as np
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

# Define functions for Volterra, Transmitter, and Receiver models
def volterra(x, h, H):
    T = len(x)
    y_t = torch.zeros(T, dtype=torch.double, device=device)
    for i in range(len(h)):
        y_t[i:] += h[i] * x[:T-i]
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            valid_length = T - max(i, j)
            y_t[max(i, j):valid_length + max(i, j)] += H[i, j] * x[:valid_length] * x[:valid_length]
    return y_t

def forward_pass(tx_symbols_input, channel, receiver_rx, h, H, padding):
    # Upsample
    tx_symbols_up = torch.zeros((tx_symbols_input.numel() * SPS,), dtype=torch.double, device=device)
    tx_symbols_up[0::SPS] = tx_symbols_input.double()

    # Apply Volterra series
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

def train_volterra_model(tx_symbols, receiver_rx, h, H, optimizer, channel, num_epochs, batch_size):
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

def train_model(tx_symbols, network, receiver_rx, optimizer, channel, num_epochs, batch_size, sps):
    for epoch in range(num_epochs):
        permutation = torch.randperm(tx_symbols.size(0))
        total_loss = 0
        for i in range(0, tx_symbols.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_tx_symbols = tx_symbols[indices].double().to(device)
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

def train_receiver_model(tx_symbols, tx_pulse, network, optimizer, channel, num_epochs, batch_size, sps):
    for epoch in range(num_epochs):
        permutation = torch.randperm(tx_symbols.size(0))
        total_loss = 0
        for i in range(0, tx_symbols.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_tx_symbols = tx_symbols[indices].double().to(device)
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

def train_combined_model(tx_symbols, network_tx, network_rx, optimizer, channel, num_epochs, batch_size, sps):
    for epoch in range(num_epochs):
        permutation = torch.randperm(tx_symbols.size(0))
        total_loss = 0
        for i in range(0, tx_symbols.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_tx_symbols = tx_symbols[indices].double().to(device)
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

import time

# Function to perform multiple runs and average results with timing
def run_simulation():
    num_epochs = 5
    batch_size = 512
    num_runs = 5
    SNR = 20

    # Ranges for different models
    h_H_sizes = range(4, 45, 4)
    filter_length = 64
    num_filters_range = range(1, 17, 2)
    num_filters_range_complex = range(1, 6)
    num_filters_range_combined = range(1, 9)
    num_filters_range_complex_combined = range(1, 4)

    # ser_results_volterra = []
    ser_results_tx = []
    ser_results_tx_complex = []
    ser_results_rx = []
    ser_results_rx_complex = []
    # ser_results_combined = []
    # ser_results_combined_complex = []

    # time_results_volterra = []
    time_results_tx = []
    time_results_tx_complex = []
    time_results_rx = []
    time_results_rx_complex = []
    # time_results_combined = []
    # time_results_combined_complex = []

    # Volterra model
    # for size in h_H_sizes:
    #     ser_avg = 0.0
    #     time_avg = 0.0
    #     for run in range(num_runs):
    #         print(f"Volterra - h_size: {size}, H_size: {size}, Run: {run + 1}")
    #         h = nn.Parameter(torch.zeros(size, dtype=torch.double, device=device) * 0.01)
    #         H = nn.Parameter(torch.zeros(size, size, dtype=torch.double, device=device) * 0.01)
    #         optimizer = torch.optim.Adam([h, H], lr=0.001)
    #         channel = WienerHammersteinISIChannel(snr_db=SNR, pulse_energy=pulse_energy, samples_pr_symbol=SPS)
            
    #         start_time = time.time()
    #         h, H = train_volterra_model(train_symbols, pulse_rx, h, H, optimizer, channel, num_epochs, batch_size)
    #         training_time = time.time() - start_time
            
    #         SER = evaluate_volterra_model(test_symbols, pulse_rx, h, H, channel, pulse_rx.shape[0] // 2)
    #         ser_avg += SER.item()
    #         time_avg += training_time
        
    #     ser_avg /= num_runs
    #     time_avg /= num_runs
    #     num_parameters = h.numel() + H.numel()
    #     ser_results_volterra.append((num_parameters, ser_avg))
    #     time_results_volterra.append((num_parameters, time_avg))

    # Transmitter optimization model
    for num_filters in num_filters_range:
        ser_avg = 0.0
        time_avg = 0.0
        for run in range(num_runs):
            print(f"Transmitter - filter_length: {filter_length}, num_filters: {num_filters}, Run: {run + 1}")
            network = WHChannelNet(filter_length, num_filters).to(device)
            optimizer = optim.Adam(network.parameters(), lr=0.001)
            channel = WienerHammersteinISIChannel(snr_db=SNR, pulse_energy=pulse_energy, samples_pr_symbol=SPS)
            
            start_time = time.time()
            trained_network = train_model(train_symbols, network, pulse_rx, optimizer, channel, num_epochs, batch_size, SPS)
            training_time = time.time() - start_time
            
            SER = evaluate_model(test_symbols, trained_network, pulse_rx, channel, SPS)
            ser_avg += SER.item()
            time_avg += training_time
        
        ser_avg /= num_runs
        time_avg /= num_runs
        num_parameters = count_parameters(trained_network)
        ser_results_tx.append((num_parameters, ser_avg))
        time_results_tx.append((num_parameters, time_avg))

    # Transmitter optimization model with WHChannelComplex
    for num_filters in num_filters_range_complex:
        ser_avg = 0.0
        time_avg = 0.0
        for run in range(num_runs):
            print(f"Transmitter (WHChannelComplex) - filter_length: {filter_length}, num_filters: {num_filters}, Run: {run + 1}")
            network = WHChannelNetComplex(filter_length, num_filters).to(device)
            optimizer = optim.Adam(network.parameters(), lr=0.001)
            channel = WienerHammersteinISIChannel(snr_db=SNR, pulse_energy=pulse_energy, samples_pr_symbol=SPS)
            
            start_time = time.time()
            trained_network = train_model(train_symbols, network, pulse_rx, optimizer, channel, num_epochs, batch_size, SPS)
            training_time = time.time() - start_time
            
            SER = evaluate_model(test_symbols, trained_network, pulse_rx, channel, SPS)
            ser_avg += SER.item()
            time_avg += training_time
        
        ser_avg /= num_runs
        time_avg /= num_runs
        num_parameters = count_parameters(trained_network)
        ser_results_tx_complex.append((num_parameters, ser_avg))
        time_results_tx_complex.append((num_parameters, time_avg))

    # Receiver optimization model
    for num_filters in num_filters_range:
        ser_avg = 0.0
        time_avg = 0.0
        for run in range(num_runs):
            print(f"Receiver - filter_length: {filter_length}, num_filters: {num_filters}, Run: {run + 1}")
            network = WHChannelNet(filter_length, num_filters).to(device)
            optimizer = optim.Adam(network.parameters(), lr=0.001)
            channel = WienerHammersteinISIChannel(snr_db=SNR, pulse_energy=pulse_energy, samples_pr_symbol=SPS)
            
            start_time = time.time()
            trained_network = train_receiver_model(train_symbols, pulse_tx, network, optimizer, channel, num_epochs, batch_size, SPS)
            training_time = time.time() - start_time
            
            SER = evaluate_receiver_model(test_symbols, pulse_tx, trained_network, channel, SPS)
            ser_avg += SER.item()
            time_avg += training_time
        
        ser_avg /= num_runs
        time_avg /= num_runs
        num_parameters = count_parameters(trained_network)
        ser_results_rx.append((num_parameters, ser_avg))
        time_results_rx.append((num_parameters, time_avg))

    # Receiver optimization model with WHChannelComplex
    for num_filters in num_filters_range_complex:
        ser_avg = 0.0
        time_avg = 0.0
        for run in range(num_runs):
            print(f"Receiver (WHChannelComplex) - filter_length: {filter_length}, num_filters: {num_filters}, Run: {run + 1}")
            network = WHChannelNetComplex(filter_length, num_filters).to(device)
            optimizer = optim.Adam(network.parameters(), lr=0.001)
            channel = WienerHammersteinISIChannel(snr_db=SNR, pulse_energy=pulse_energy, samples_pr_symbol=SPS)
            
            start_time = time.time()
            trained_network = train_receiver_model(train_symbols, pulse_tx, network, optimizer, channel, num_epochs, batch_size, SPS)
            training_time = time.time() - start_time
            
            SER = evaluate_receiver_model(test_symbols, pulse_tx, trained_network, channel, SPS)
            ser_avg += SER.item()
            time_avg += training_time
        
        ser_avg /= num_runs
        time_avg /= num_runs
        num_parameters = count_parameters(trained_network)
        ser_results_rx_complex.append((num_parameters, ser_avg))
        time_results_rx_complex.append((num_parameters, time_avg))

    # Combined optimization model
    # for num_filters in num_filters_range_combined:
    #     ser_avg = 0.0
    #     time_avg = 0.0
    #     for run in range(num_runs):
    #         print(f"Combined - filter_length: {filter_length}, num_filters: {num_filters}, Run: {run + 1}")
    #         network_tx = WHChannelNet(filter_length, num_filters).to(device)
    #         network_rx = WHChannelNet(filter_length, num_filters).to(device)
    #         optimizer = optim.Adam(list(network_tx.parameters()) + list(network_rx.parameters()), lr=0.001)
    #         channel = WienerHammersteinISIChannel(snr_db=SNR, pulse_energy=pulse_energy, samples_pr_symbol=SPS)
            
    #         start_time = time.time()
    #         trained_network_tx, trained_network_rx = train_combined_model(train_symbols, network_tx, network_rx, optimizer, channel, num_epochs, batch_size, SPS)
    #         training_time = time.time() - start_time
            
    #         SER = evaluate_combined_model(test_symbols, trained_network_tx, trained_network_rx, channel, SPS)
    #         ser_avg += SER.item()
    #         time_avg += training_time
        
    #     ser_avg /= num_runs
    #     time_avg /= num_runs
    #     num_parameters = count_parameters(trained_network_tx) + count_parameters(trained_network_rx)
    #     ser_results_combined.append((num_parameters, ser_avg))
    #     time_results_combined.append((num_parameters, time_avg))

    # Combined optimization model with WHChannelComplex
    # for num_filters in num_filters_range_complex_combined:
    #     ser_avg = 0.0
    #     time_avg = 0.0
    #     for run in range(num_runs):
    #         print(f"Combined (WHChannelComplex) - filter_length: {filter_length}, num_filters: {num_filters}, Run: {run + 1}")
    #         network_tx = WHChannelNetComplex(filter_length, num_filters).to(device)
    #         network_rx = WHChannelNetComplex(filter_length, num_filters).to(device)
    #         optimizer = optim.Adam(list(network_tx.parameters()) + list(network_rx.parameters()), lr=0.001)
    #         channel = WienerHammersteinISIChannel(snr_db=SNR, pulse_energy=pulse_energy, samples_pr_symbol=SPS)
            
    #         start_time = time.time()
    #         trained_network_tx, trained_network_rx = train_combined_model(train_symbols, network_tx, network_rx, optimizer, channel, num_epochs, batch_size, SPS)
    #         training_time = time.time() - start_time
            
    #         SER = evaluate_combined_model(test_symbols, trained_network_tx, trained_network_rx, channel, SPS)
    #         ser_avg += SER.item()
    #         time_avg += training_time
        
    #     ser_avg /= num_runs
    #     time_avg /= num_runs
    #     num_parameters = count_parameters(trained_network_tx) + count_parameters(trained_network_rx)
    #     ser_results_combined_complex.append((num_parameters, ser_avg))
    #     time_results_combined_complex.append((num_parameters, time_avg))

    # Plotting results

    # SER vs Number of Parameters
    plt.figure(figsize=(12, 8))

    # Volterra
    # num_params_volterra, ser_volterra = zip(*ser_results_volterra)
    # plt.plot(num_params_volterra, ser_volterra, marker='o', linestyle='-', label="Volterra")

    # Transmitter
    num_params_tx, ser_tx = zip(*ser_results_tx)
    plt.plot(num_params_tx, ser_tx, marker='d', linestyle='-', label="Transmitter Optimization")

    # Transmitter (WHChannelComplex)
    num_params_tx_complex, ser_tx_complex = zip(*ser_results_tx_complex)
    plt.plot(num_params_tx_complex, ser_tx_complex, marker='d', linestyle='--', label="Transmitter Optimization (WHChannelComplex)")

    # Receiver
    num_params_rx, ser_rx = zip(*ser_results_rx)
    plt.plot(num_params_rx, ser_rx, marker='s', linestyle='-', label="Receiver Optimization")

    # Receiver (WHChannelComplex)
    num_params_rx_complex, ser_rx_complex = zip(*ser_results_rx_complex)
    plt.plot(num_params_rx_complex, ser_rx_complex, marker='s', linestyle='--', label="Receiver Optimization (WHChannelComplex)")

    # Combined
    # num_params_combined, ser_combined = zip(*ser_results_combined)
    # plt.plot(num_params_combined, ser_combined, marker='x', linestyle='-', label="Combined Optimization")

    # # Combined (WHChannelComplex)
    # num_params_combined_complex, ser_combined_complex = zip(*ser_results_combined_complex)
    # plt.plot(num_params_combined_complex, ser_combined_complex, marker='x', linestyle='--', label="Combined Optimization (WHChannelComplex)")

    plt.xlabel("Number of Parameters")
    plt.ylabel("Symbol Error Rate (SER)")
    plt.yscale("log")
    plt.title(f"SER vs. Number of Parameters for Different Models at SNR = {SNR}")
    plt.legend(ncol=2)
    plt.grid(True)
    plt.show()

    # Training Time vs Number of Parameters
    plt.figure(figsize=(12, 8))
    
    # Volterra
    # num_params_volterra, time_volterra = zip(*time_results_volterra)
    # plt.plot(num_params_volterra, time_volterra, marker='o', linestyle='-', label="Volterra")

    # Transmitter
    num_params_tx, time_tx = zip(*time_results_tx)
    plt.plot(num_params_tx, time_tx, marker='d', linestyle='-', label="Transmitter Optimization")

    # Transmitter (WHChannelComplex)
    num_params_tx_complex, time_tx_complex = zip(*time_results_tx_complex)
    plt.plot(num_params_tx_complex, time_tx_complex, marker='d', linestyle='--', label="Transmitter Optimization (WHChannelComplex)")

    # Receiver
    num_params_rx, time_rx = zip(*time_results_rx)
    plt.plot(num_params_rx, time_rx, marker='s', linestyle='-', label="Receiver Optimization")

    # Receiver (WHChannelComplex)
    num_params_rx_complex, time_rx_complex = zip(*time_results_rx_complex)
    plt.plot(num_params_rx_complex, time_rx_complex, marker='s', linestyle='--', label="Receiver Optimization (WHChannelComplex)")

    # Combined
    # num_params_combined, time_combined = zip(*time_results_combined)
    # plt.plot(num_params_combined, time_combined, marker='x', linestyle='-', label="Combined Optimization")

    # # Combined (WHChannelComplex)
    # num_params_combined_complex, time_combined_complex = zip(*time_results_combined_complex)
    # plt.plot(num_params_combined_complex, time_combined_complex, marker='x', linestyle='--', label="Combined Optimization (WHChannelComplex)")

    plt.xlabel("Number of Parameters")
    plt.ylabel("Training Time (s)")
    plt.title(f"Training Time vs. Number of Parameters for Different Models at SNR = {SNR}")
    plt.legend(ncol=2)
    plt.grid(True)
    plt.show()

# Run the simulation
run_simulation()