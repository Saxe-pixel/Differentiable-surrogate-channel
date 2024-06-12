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
N_SYMBOLS = int(2 * 1e5)
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
pulse_energy = np.max(gg)

class ReceiverNet(nn.Module):
    def __init__(self, filter_length, num_filters=8):
        super(ReceiverNet, self).__init__()
        self.conv1 = nn.Conv1d(1, num_filters, filter_length, padding=filter_length // 2, bias=False, dtype=torch.double)
        self.conv2 = nn.Conv1d(num_filters, 1, filter_length, padding=filter_length // 2, bias=False, dtype=torch.double)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x

# Training Function
def train_receiver_net(tx_symbols, network, transmitter_tx, optimizer, channel, num_epochs, batch_size, sps):
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

            # Pass through the fixed transmitter filter
            tx = torch.nn.functional.conv1d(tx_symbols_up.view(1, 1, -1), transmitter_tx.view(1, 1, -1), padding=transmitter_tx.shape[0] // 2).squeeze()

            # Pass through the actual WH channel
            y = channel.forward(tx)

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

        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / (i // batch_size + 1)}")
    return network

# Evaluation Function
def evaluate_model(tx_symbols_input, network, transmitter_tx, channel, sps):
    with torch.no_grad():
        # Prepare input signal
        tx_symbols_up = torch.zeros((tx_symbols_input.numel() * sps,), dtype=torch.double).to(device)
        tx_symbols_up[0::sps] = tx_symbols_input.double()

        # Pass through the fixed transmitter filter
        tx = torch.nn.functional.conv1d(tx_symbols_up.view(1, 1, -1), transmitter_tx.view(1, 1, -1), padding=transmitter_tx.shape[0] // 2).squeeze()

        # Forward pass through the actual WH channel
        y = channel.forward(tx)

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
        print(f"Evaluation: Symbol Error Rate: {SER.item()}")
        return SER

# Theoretical SER Calculation
def theoretical_ser(snr_db, pulse_energy, modulation_order):
    log2M = np.log2(modulation_order)
    SNR_linear = 10 ** (snr_db / 10)
    Eb_N0_linear = SNR_linear / log2M
    Q = lambda x: 0.5 * erfc(x / np.sqrt(2))
    SER_theoretical = 2 * (1 - 1 / modulation_order) * Q(np.sqrt(2 * log2M * Eb_N0_linear))
    return SER_theoretical

# Initialize receiver network and optimizer
receiver_net = ReceiverNet(FILTER_LENGTH).to(device)
optimizer = optim.Adam(receiver_net.parameters(), lr=0.001)

# SNR settings and results storage
SNRs = range(0, 9)
num_epochs = 10
batch_size = 512

theoretical_SERs = [theoretical_ser(snr_db, pulse_energy, 4) for snr_db in SNRs]
wh_isi_SERs = []

# Initialize the actual WH channel
wiener_hammerstein_channel = WienerHammersteinISIChannel(snr_db=10, pulse_energy=pulse_energy, samples_pr_symbol=SPS)

# Simulation loop
for snr_db in SNRs:
    print(f"Training and evaluating at SNR: {snr_db} dB")
    wiener_hammerstein_channel = WienerHammersteinISIChannel(snr_db=snr_db, pulse_energy=pulse_energy, samples_pr_symbol=SPS)
    
    # Train the receiver network using the actual WH channel
    trained_network_wh = train_receiver_net(train_symbols, receiver_net, pulse_tx, optimizer, wiener_hammerstein_channel, num_epochs, batch_size, SPS)
    ser_wh = evaluate_model(test_symbols, trained_network_wh, pulse_tx, wiener_hammerstein_channel, SPS)
    wh_isi_SERs.append(ser_wh)

# Plotting results
plt.figure(figsize=(12, 8))
plt.plot(SNRs, theoretical_SERs, label="Theoretical SER", marker='o')
plt.plot(SNRs, wh_isi_SERs, label="WienerHammerstein ISI Channel SER", marker='d')
plt.xlabel("SNR (dB)")
plt.ylabel("Symbol Error Rate (SER)")
plt.yscale("log")
plt.title("Symbol Error Rate Across Different Channels")
plt.legend()
plt.grid(True)
plt.show()
