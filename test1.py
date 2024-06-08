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
tx_symbols = torch.from_numpy(random_obj.choice(pam_symbols, size=(N_SYMBOLS,), replace=True)).to(device)

# Split the data into training and testing
train_size = int(0.7 * N_SYMBOLS)
test_size = N_SYMBOLS - train_size
train_symbols = tx_symbols[:train_size]
test_symbols = tx_symbols[train_size:]

# Construct pulse shape for both transmitter and receiver
t, g = rrcosfilter(FILTER_LENGTH, 0.5, Ts, fs)
g /= np.linalg.norm(g)  # Normalize pulse to have unit energy
gg = np.convolve(g, g[::-1])
pulse_rx = torch.from_numpy(g).double().to(device)  # Receiver pulse (fixed)
pulse_energy = np.max(gg)

# Define the pulse for the transmitter (to be optimized)
pulse_tx = torch.zeros((FILTER_LENGTH,), dtype=torch.double).requires_grad_(True)

# Define the pulse for the receiver (fixed)
pulse_rx = torch.from_numpy(g).double()  # No requires_grad_() as it's not being optimized

# Define training and evaluation functions
def forward_pass(tx_symbols_input, optimized_pulse, channel, reciever_rx, padding):
    tx_symbols_up = torch.zeros((tx_symbols_input.numel() * SPS,), dtype=torch.double)
    tx_symbols_up[0::SPS] = tx_symbols_input.double()
    x = F.conv1d(tx_symbols_up.view(1, 1, -1), optimized_pulse.view(1, 1, -1), padding=padding)
    y = channel.forward(x.squeeze())
    rx = F.conv1d(y.view(1, 1, -1), reciever_rx.view(1, 1, -1).flip(dims=[2]), padding=padding).squeeze()
    delay = estimate_delay(rx, SPS)
    rx = rx[delay::SPS]
    rx = rx[:tx_symbols_input.numel()]
    return rx, tx_symbols_up

# Neural Network for Pulse Shaping
import torch.nn as nn

class PulseShapeNet(nn.Module):
    def __init__(self, filter_length):
        super(PulseShapeNet, self).__init__()
        self.pulse = nn.Parameter(torch.randn(filter_length, dtype=torch.float64, requires_grad=True))
        
    def forward(self, x):
        # x is the upsampled signal
        return torch.nn.functional.conv1d(x.view(1, 1, -1), self.pulse.view(1, 1, -1), padding=self.pulse.shape[0] // 2)

# class PulseShapeNet(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(PulseShapeNet, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x

def train_model(tx_symbols, network, receiver_rx, optimizer, channel, num_epochs, batch_size, sps):
    for epoch in range(num_epochs):
        permutation = torch.randperm(tx_symbols.size(0))
        total_loss = 0
        for i in range(0, tx_symbols.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_tx_symbols = tx_symbols[indices].double().to(device)

            optimizer.zero_grad()

            # Upsample input symbols
            tx_symbols_up = torch.zeros((batch_tx_symbols.numel() * sps,), dtype=torch.double).to(device)
            tx_symbols_up[0::sps] = batch_tx_symbols

            # Forward pass through the network (pulse shaping)
            shaped_pulse = network(tx_symbols_up.view(1, 1, -1))

            # Pass through the channel
            y = channel.forward(shaped_pulse.squeeze())
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



def evaluate_model(tx_symbols_input, network, receiver_rx, channel, sps):
    with torch.no_grad():
        # Prepare input signal
        tx_symbols_up = torch.zeros((tx_symbols_input.numel() * sps,), dtype=torch.double).to(device)
        tx_symbols_up[0::sps] = tx_symbols_input.double()

        # Forward pass through the network and channel
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



def theoretical_ser(snr_db, pulse_energy, modulation_order):
    log2M = np.log2(modulation_order)
    SNR_linear = 10 ** (snr_db / 10)
    Eb_N0_linear = SNR_linear / log2M
    Q = lambda x: 0.5 * erfc(x / np.sqrt(2))
    SER_theoretical = 2 * (1 - 1 / modulation_order) * Q(np.sqrt(2 * log2M * Eb_N0_linear))
    return SER_theoretical




# Initialize network and optimizer
pulse_shaper_net = PulseShapeNet(FILTER_LENGTH).double()
optimizer = optim.Adam(pulse_shaper_net.parameters(), lr=0.001)

# SNR settings and results storage
SNRs = range(0, 9)
num_epochs = 15
batch_size = 1024

theoretical_SERs = [theoretical_ser(snr_db, pulse_energy, 4) for snr_db in SNRs]
awgn_SERs = []
awgn_isi_SERs = []
wh_isi_SERs = []

# Simulation loop
for snr_db in SNRs:
    print(f"Training and evaluating at SNR: {snr_db} dB")
    
    # AWGN Channel
    awgn_channel = AWGNChannel(snr_db=snr_db, pulse_energy=pulse_energy)
    trained_network_awgn = train_model(train_symbols, pulse_shaper_net, pulse_rx, optimizer, awgn_channel, num_epochs, batch_size, SPS)
    ser_awgn = evaluate_model(test_symbols, trained_network_awgn, pulse_rx, awgn_channel, SPS)
    awgn_SERs.append(ser_awgn)
    
    # AWGN with Linear ISI Channel
    awgn_isi_channel = AWGNChannelWithLinearISI(snr_db=snr_db, pulse_energy=pulse_energy, samples_pr_symbol=SPS)
    trained_network_awgn_isi = train_model(train_symbols, pulse_shaper_net, pulse_rx, optimizer, awgn_isi_channel, num_epochs, batch_size, SPS)
    ser_awgn_isi = evaluate_model(test_symbols, trained_network_awgn_isi, pulse_rx, awgn_isi_channel, SPS)
    awgn_isi_SERs.append(ser_awgn_isi)
    
    # Wiener-Hammerstein ISI Channel
    wh_channel = WienerHammersteinISIChannel(snr_db=snr_db, pulse_energy=pulse_energy, samples_pr_symbol=SPS, dtype=torch.float64)
    trained_network_wh = train_model(train_symbols, pulse_shaper_net, pulse_rx, optimizer, wh_channel, num_epochs, batch_size, SPS)
    ser_wh = evaluate_model(test_symbols, trained_network_wh, pulse_rx, wh_channel, SPS)
    wh_isi_SERs.append(ser_wh)

# Plotting results
plt.figure(figsize=(12, 8))
plt.plot(SNRs, theoretical_SERs, label="Theoretical SER", marker='o')
plt.plot(SNRs, awgn_SERs, label="AWGN Channel SER", marker='x')
plt.plot(SNRs, awgn_isi_SERs, label="AWGN with ISI Channel SER", marker='s')
plt.plot(SNRs, wh_isi_SERs, label="WienerHammerstein ISI Channel SER", marker='d')
plt.xlabel("SNR (dB)")
plt.ylabel("Symbol Error Rate (SER)")
plt.yscale("log")
plt.title("Symbol Error Rate Across Different Channels")
plt.legend()
plt.grid(True)
plt.show()