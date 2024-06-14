import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from commpy.filters import rrcosfilter
from lib.utility import estimate_delay, find_closest_symbol
from lib.channels import AWGNChannel, AWGNChannelWithLinearISI, WienerHammersteinISIChannel
from scipy.special import erfc

# Define device and simulation parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 12345
N_SYMBOLS = int(5 * 1e5)
SPS = 8  # Samples-per-symbol (oversampling rate)
BAUD_RATE = 10e6  # Number of symbols transmitted per second
FILTER_LENGTH = 64
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
t, g = rrcosfilter(FILTER_LENGTH, 0.5, Ts, fs)
g /= np.linalg.norm(g)  # Normalize pulse to have unit energy
gg = np.convolve(g, g[::-1])
pulse_tx = torch.from_numpy(g).double().to(device)  # Transmitter pulse (fixed)
pulse_rx = torch.from_numpy(g).double().to(device)  # Receiver pulse (fixed)
pulse_energy = np.max(gg)

class ChannelNet(nn.Module):
    def __init__(self, filter_length, num_filters=8, initial_non_linear_coefficients=(0,0,0)):
        super(ChannelNet, self).__init__()
        self.conv1 = nn.Conv1d(1, num_filters, filter_length, padding=filter_length // 2, bias=False, dtype=torch.double)
        self.conv2 = nn.Conv1d(num_filters, 1, filter_length, padding=filter_length // 2, bias=False, dtype=torch.double)
        
        # Making non-linear coefficients learnable parameters
        self.a0 = nn.Parameter(torch.tensor(initial_non_linear_coefficients[0], dtype=torch.double))
        self.a1 = nn.Parameter(torch.tensor(initial_non_linear_coefficients[1], dtype=torch.double))
        self.a2 = nn.Parameter(torch.tensor(initial_non_linear_coefficients[2], dtype=torch.double))

        # Initializing the weights of the convolutional layers randomly
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.a0 * x + self.a1 * x ** 2 + self.a2 * x ** 3
        x = self.conv2(x)
        return x

# Training Function for Transmitter Optimization
def train_pulse_shaping_net(tx_symbols, network, receiver_rx, optimizer, channel, num_epochs, batch_size, sps):
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

            # Forward pass through the network (pulse shaping)
            shaped_pulse = network(tx_symbols_up.view(1, 1, -1))

            # Pass through the actual WH channel
            y = channel.forward(shaped_pulse.squeeze())

            # Pass through the receiver filter
            rx = torch.nn.functional.conv1d(y.view(1, 1, -1), receiver_rx.view(1, 1, -1).flip(dims=[2]), padding=receiver_rx.shape[0] // 2).squeeze()

            # Delay estimation and synchronization
            delay = estimate_delay(rx, sps)
            rx = rx[delay::sps]
            rx = rx[:batch_tx_symbols.numel()]

            # Loss calculation and backpropagation
            loss = torch.nn.functional.mse_loss(rx, batch_tx_symbols)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / (i // batch_size + 1)}")
    return network

# Training Function for Receiver Optimization
def train_receiver_net(tx_symbols, tx_network, rx_network, optimizer, channel, num_epochs, batch_size, sps):
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
            shaped_pulse = torch.nn.functional.conv1d(tx_symbols_up.view(1, 1, -1), tx_network.view(1, 1, -1), padding=tx_network.shape[0] // 2)

            # Pass through the actual WH channel
            y = channel.forward(shaped_pulse.squeeze())

            # Forward pass through the receiver network
            rx = rx_network(y.view(1, 1, -1)).squeeze()

            # Delay estimation and synchronization
            delay = estimate_delay(rx, sps)
            rx = rx[delay::sps]
            rx = rx[:batch_tx_symbols.numel()]

            # Loss calculation and backpropagation
            loss = torch.nn.functional.mse_loss(rx, batch_tx_symbols)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / (i // batch_size + 1)}")
    return rx_network

# Evaluation Function for Transmitter Optimization
def evaluate_model(tx_symbols_input, network, receiver_rx, channel, sps):
    with torch.no_grad():
        # Prepare input signal
        tx_symbols_up = torch.zeros((tx_symbols_input.numel() * sps,), dtype=torch.double).to(device)
        tx_symbols_up[0::sps] = tx_symbols_input.double()

        # Forward pass through the network and actual WH channel
        shaped_pulse = network(tx_symbols_up.view(1, 1, -1))
        y = channel.forward(shaped_pulse.squeeze())
        rx_eval = torch.nn.functional.conv1d(y.view(1, 1, -1), receiver_rx.view(1, 1, -1).flip(dims=[2]), padding=receiver_rx.shape[0] // 2).squeeze()

        # Delay estimation and synchronization
        delay = estimate_delay(rx_eval, sps)
        symbols_est = rx_eval[delay::sps][:tx_symbols_input.numel()]

        # Symbol decision
        symbols_est = find_closest_symbol(symbols_est, torch.from_numpy(pam_symbols))

        # Error calculation
        error = torch.sum(torch.logical_not(torch.eq(symbols_est, tx_symbols_input)))
        SER = error.float() / tx_symbols_input.numel()
        return SER

# Evaluation Function for Receiver Optimization
def evaluate_receiver_model(tx_symbols_input, tx_network, rx_network, channel, sps):
    with torch.no_grad():
        # Prepare input signal
        tx_symbols_up = torch.zeros((tx_symbols_input.numel() * sps,), dtype=torch.double).to(device)
        tx_symbols_up[0::sps] = tx_symbols_input.double()

        # Transmit through the fixed pulse shaping filter
        shaped_pulse = torch.nn.functional.conv1d(tx_symbols_up.view(1, 1, -1), tx_network.view(1, 1, -1), padding=tx_network.shape[0] // 2)

        # Pass through the actual WH channel
        y = channel.forward(shaped_pulse.squeeze())

        # Forward pass through the receiver network
        rx_eval = rx_network(y.view(1, 1, -1)).squeeze()

        # Delay estimation and synchronization
        delay = estimate_delay(rx_eval, sps)
        symbols_est = rx_eval[delay::sps][:tx_symbols_input.numel()]

        # Symbol decision
        symbols_est = find_closest_symbol(symbols_est, torch.from_numpy(pam_symbols))

        # Error calculation
        error = torch.sum(torch.logical_not(torch.eq(symbols_est, tx_symbols_input)))
        SER = error.float() / tx_symbols_input.numel()
        return SER

# Function to perform multiple runs and average results
def run_simulation():
    # Fixed SNR value
    fixed_snr_db = 20
    num_epochs = 5
    batch_size = 512

    # Range of non-linear coefficients to test
    xxx_values = np.linspace(0, 1, 10)
    ser_results_tx = np.zeros(len(xxx_values))
    ser_results_rx = np.zeros(len(xxx_values))
    
    num_runs = 5

    for run in range(num_runs):
        print(f"Run {run + 1} / {num_runs}")
        for idx, xxx in enumerate(xxx_values):
            print(f"  Testing non-linear coefficient a1 = {xxx}")
            wiener_hammerstein_channel = WienerHammersteinISIChannel(snr_db=fixed_snr_db, pulse_energy=pulse_energy, samples_pr_symbol=SPS, non_linear_coefficients=(1.0, xxx, 0))

            # Reinitialize network and optimizer for transmitter training
            network_tx = ChannelNet(FILTER_LENGTH).to(device)
            optimizer_tx = optim.Adam(network_tx.parameters(), lr=0.001)
            
            # Train the pulse shaping network using the actual WH channel
            trained_network_tx = train_pulse_shaping_net(train_symbols, network_tx, pulse_rx, optimizer_tx, wiener_hammerstein_channel, num_epochs, batch_size, SPS)
            ser_tx = evaluate_model(test_symbols, trained_network_tx, pulse_rx, wiener_hammerstein_channel, SPS)
            ser_results_tx[idx] += ser_tx.item()
            print(f"    SER for a1 = {xxx} (Transmitter Optimization): {ser_tx.item()}")
            
            # Reinitialize network and optimizer for receiver training
            network_rx = ChannelNet(FILTER_LENGTH).to(device)
            optimizer_rx = optim.Adam(network_rx.parameters(), lr=0.001)

            # Train the receiver network using the actual WH channel
            trained_network_rx = train_receiver_net(train_symbols, pulse_tx, network_rx, optimizer_rx, wiener_hammerstein_channel, num_epochs, batch_size, SPS)
            ser_rx = evaluate_receiver_model(test_symbols, pulse_tx, trained_network_rx, wiener_hammerstein_channel, SPS)
            ser_results_rx[idx] += ser_rx.item()
            print(f"    SER for a1 = {xxx} (Receiver Optimization): {ser_rx.item()}")

    # Calculate average SER results
    ser_results_tx /= num_runs
    ser_results_rx /= num_runs

    # Plotting results
    plt.figure(figsize=(12, 8))
    plt.plot(xxx_values, ser_results_tx, label="Transmitter Optimization SER", marker='d')
    plt.plot(xxx_values, ser_results_rx, label="Receiver Optimization SER", marker='o')
    plt.xlabel("Non-linear coefficient")
    plt.ylabel("Symbol Error Rate (SER)")
    plt.yscale("log")
    plt.title("Average Symbol Error Rate for different Non-linear Coefficients at SNR = 20 dB")
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the simulation
run_simulation()