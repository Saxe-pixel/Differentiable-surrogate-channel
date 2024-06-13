import numpy as np
import torch
import matplotlib.pyplot as plt
import torchaudio.functional as taf
from commpy.filters import rrcosfilter
import torch.nn.functional as F
from lib.utility import estimate_delay, find_closest_symbol
from lib.channels import AWGNChannel, AWGNChannelWithLinearISI
import torch.optim as optim
from lib.utility import estimate_delay, find_closest_symbol
from lib.channels import AWGNChannel, AWGNChannelWithLinearISI, WienerHammersteinISIChannel
import torch.nn as nn
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"




print(device)
# Simulation parameters
SEED = 12345
N_SYMBOLS = int(2 * 1e5)
SPS = 8  # samples-per-symbol (oversampling rate)
SNR_DB = 8.0  # signal-to-noise ratio in dB
BAUD_RATE = 10e6  # number of symbols transmitted per second
FILTER_LENGTH = 64
Ts = 1 / BAUD_RATE  # symbol length
fs = BAUD_RATE * SPS
random_obj = np.random.default_rng(SEED)

# Generate data - use Pulse Amplitude Modulation (PAM)
pam_symbols = np.array([-3, -1, 1, 3])
tx_symbols = torch.from_numpy(random_obj.choice(pam_symbols, size=(N_SYMBOLS,), replace=True)).to(device)

# Split the data into training and testing
train_size = int(0.7 * N_SYMBOLS)
test_size = N_SYMBOLS - train_size
train_symbols = tx_symbols[:train_size]
test_symbols = tx_symbols[train_size:]

# Construct pulse shape for the receiver
t, g = rrcosfilter(FILTER_LENGTH, 0.5, Ts, fs)
g /= np.linalg.norm(g)  # Normalize pulse to have unit energy
gg = np.convolve(g, g[::-1])
pulse_energy = np.max(gg)

# Define the pulse for the receiver (fixed)
pulse_rx = torch.from_numpy(g).double().to(device)

# Define h and H for Volterra series
h_size = 7
H_size = 7
h = nn.Parameter(torch.randn(h_size, dtype=torch.double, device=device))
H = nn.Parameter(torch.randn(H_size, H_size, dtype=torch.double, device=device))

optimizer = torch.optim.Adam([h, H], lr=0.01)
channel = WienerHammersteinISIChannel(snr_db=SNR_DB, pulse_energy=pulse_energy, samples_pr_symbol=SPS)

# Move channel filters to the correct device and data type
channel.isi_filter1 = channel.isi_filter1.to(device=device, dtype=torch.double)
channel.isi_filter2 = channel.isi_filter2.to(device=device, dtype=torch.double)

# Calculate padding to achieve "same" output length
sym_trim = FILTER_LENGTH // 2 // SPS
num_epochs = 5  # Number of iterations for optimization
batch_size = 512  # Batch size for optimization

def volterra(x, h, H):
    T = len(x)
    y_t = torch.zeros(T, dtype=torch.double, device=device)
    
    # First summation (vectorized)
    for i in range(len(h)):
        y_t[i:] += h[i] * x[:T-i]

    # Second summation (vectorized)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            valid_length = T - max(i, j)
            y_t[max(i, j):valid_length + max(i, j)] += H[i, j] * x[:valid_length] * x[:valid_length]

    return y_t

def forward_pass(tx_symbols_input, channel, reciever_rx, h, H, padding):
    # Upsample
    tx_symbols_up = torch.zeros((tx_symbols_input.numel() * SPS,), dtype=torch.double, device=device)
    tx_symbols_up[0::SPS] = tx_symbols_input.double()

    # Apply Volterra series
    x = volterra(tx_symbols_up, h, H)

    # Simulate the channel
    y = channel.forward(x)

    # Apply receiver with consistent padding
    rx = F.conv1d(y.view(1, 1, -1), reciever_rx.view(1, 1, -1).flip(dims=[2]), padding=padding).squeeze()
    delay = estimate_delay(rx, SPS)
    rx = rx[delay::SPS]
    rx = rx[:tx_symbols_input.numel()]
    return rx, tx_symbols_up

def train_model(tx_symbols, reciever_rx, h, H, optimizer, channel, num_epochs, batch_size):
    for epoch in range(num_epochs):
        permutation = torch.randperm(train_size, device=device)
        for i in range(0, train_size, batch_size):
            indices = permutation[i:i+batch_size]
            batch_tx_symbols = tx_symbols[indices].double()

            optimizer.zero_grad()
            rx, _ = forward_pass(batch_tx_symbols, channel, reciever_rx, h, H, reciever_rx.shape[0] // 2)

            # Compute loss
            min_length = min(len(rx), len(batch_tx_symbols))
            rx = rx[:min_length]
            batch_tx_symbols = batch_tx_symbols[:min_length]

            loss = F.mse_loss(rx, batch_tx_symbols)  # Mean squared error loss
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    return h, H

def evaluate_model(tx_symbols_input, reciever_rx, h, H, channel, padding):
    with torch.no_grad():
        tx_symbols_eval = torch.zeros((tx_symbols_input.numel() * SPS,), dtype=torch.double, device=device)
        tx_symbols_eval[0::SPS] = tx_symbols_input.double()

        # Apply Volterra series
        x = volterra(tx_symbols_eval, h, H)

        # Simulate the channel
        y = channel.forward(x)

        # Apply receiver with consistent padding
        rx_eval = F.conv1d(y.view(1, 1, -1), reciever_rx.view(1, 1, -1).flip(dims=[2]), padding=padding).squeeze()

        delay = estimate_delay(rx_eval, SPS)
        symbols_est = rx_eval[delay::SPS]
        symbols_est = symbols_est[:test_size]
        symbols_est = find_closest_symbol(symbols_est, torch.from_numpy(pam_symbols).to(device))

        # Calculate errors and SER
        error = torch.sum(torch.logical_not(torch.eq(symbols_est, tx_symbols_input)))
        SER = error.float() / len(tx_symbols_input)
        return SER

h, H = train_model(train_symbols, pulse_rx, h, H, optimizer, channel, num_epochs, batch_size)
SER = evaluate_model(test_symbols, pulse_rx, h, H, channel, pulse_rx.shape[0] // 2)
print(f"SER: {SER}")