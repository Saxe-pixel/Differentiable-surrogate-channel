import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from commpy.filters import rrcosfilter
from lib.utility import estimate_delay, find_closest_symbol
from lib.channels import AWGNChannel, AWGNChannelWithLinearISI
import torch.optim as optim
from scipy.special import erfc

# Define device and simulation parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 12345
N_SYMBOLS = int(2 * 1e5)
SPS = 8  # samples-per-symbol (oversampling rate)
BAUD_RATE = 10e6  # number of symbols transmitted pr. second
FILTER_LENGTH = 64
Ts = 1 / BAUD_RATE  # symbol length
fs = BAUD_RATE * SPS
random_obj = np.random.default_rng(SEED)

# Generate data - use Pulse Amplitude Modulation (PAM)
pam_symbols = np.array([-3, -1, 1, 3])
tx_symbols = torch.from_numpy(random_obj.choice(pam_symbols, size=(N_SYMBOLS,), replace=True))

# Split the data into training and testing
train_size = int(0.7 * N_SYMBOLS)
test_size = int(N_SYMBOLS - train_size)
train_symbols = tx_symbols[:train_size]
test_symbols = tx_symbols[train_size:]

# Construct pulse shape for both transmitter and receiver
t, g = rrcosfilter(FILTER_LENGTH, 0.5, Ts, fs)
g /= np.linalg.norm(g)  # Normalize pulse to have unit energy
gg = np.convolve(g, g[::-1])
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

def train_model(tx_symbols, optimized_pulse, reciever_rx, optimizer, channel, num_epochs, batch_size):
    for epoch in range(num_epochs):
        permutation = torch.randperm(train_size)
        for i in range(0, train_size, batch_size):
            indices = permutation[i:i + batch_size]
            batch_tx_symbols = tx_symbols[indices].double()
            optimizer.zero_grad()
            rx, _ = forward_pass(batch_tx_symbols, optimized_pulse, channel, reciever_rx, optimized_pulse.shape[0] // 2)
            rx_trimmed = rx[:batch_tx_symbols.numel()]
            loss = F.mse_loss(rx_trimmed, batch_tx_symbols)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
    return optimized_pulse

def evaluate_model(tx_symbols_input, optimized_pulse, reciever_rx, channel, padding):
    with torch.no_grad():
        tx_symbols_eval = torch.zeros((tx_symbols_input.numel() * SPS,), dtype=torch.double)
        tx_symbols_eval[0::SPS] = tx_symbols_input.double()
        x = F.conv1d(tx_symbols_eval.view(1, 1, -1), optimized_pulse.view(1, 1, -1), padding=padding)
        y = channel.forward(x.squeeze())
        rx_eval = F.conv1d(y.view(1, 1, -1), reciever_rx.view(1, 1, -1).flip(dims=[2]), padding=padding).squeeze()
        delay = estimate_delay(rx_eval, SPS)
        symbols_est = rx_eval[delay::SPS]
        symbols_est = symbols_est[:test_size]
        symbols_est = find_closest_symbol(symbols_est, torch.from_numpy(pam_symbols))
        error = torch.sum(torch.logical_not(torch.eq(symbols_est, tx_symbols_input)))
        SER = error.float() / len(tx_symbols_input)
        return SER

def theoretical_ser(snr_db, pulse_energy, modulation_order):
    log2M = np.log2(modulation_order)
    SNR_linear = 10 ** (snr_db / 10)
    Eb_N0_linear = SNR_linear / log2M
    Q = lambda x: 0.5 * erfc(x / np.sqrt(2))
    SER_theoretical = 2 * (1 - 1 / modulation_order) * Q(np.sqrt(2 * log2M * Eb_N0_linear))
    return SER_theoretical

# Plotting results
SNRs = range(0, 9)
theoretical_SERs = [theoretical_ser(snr_db, pulse_energy, 4) for snr_db in SNRs]
awgn_SERs = []
awgn_isi_SERs = []

num_epochs = 10
batch_size = 512*2
n_runs = 10

for snr_db in SNRs:
    awgn_ser_list = []
    awgn_isi_ser_list = []

    for _ in range(n_runs):
        # Train and evaluate on AWGNChannel
        pulse_tx = torch.zeros((FILTER_LENGTH,), dtype=torch.double).requires_grad_(True)
        optimizer = optim.Adam([pulse_tx], lr=0.001)
        channel = AWGNChannel(snr_db=snr_db, pulse_energy=pulse_energy)
        optimized_pulses = train_model(train_symbols, pulse_tx, pulse_rx, optimizer, channel, num_epochs, batch_size)
        SER = evaluate_model(test_symbols, optimized_pulses, pulse_rx, channel, pulse_tx.shape[0] // 2)
        awgn_ser_list.append(SER.item())

        # Train and evaluate on AWGNChannelWithLinearISI
        pulse_tx = torch.zeros((FILTER_LENGTH,), dtype=torch.double).requires_grad_(True)
        optimizer = optim.Adam([pulse_tx], lr=0.001)
        channel = AWGNChannelWithLinearISI(snr_db=snr_db, pulse_energy=pulse_energy, samples_pr_symbol=SPS)
        optimized_pulses = train_model(train_symbols, pulse_tx, pulse_rx, optimizer, channel, num_epochs, batch_size)
        SER = evaluate_model(test_symbols, optimized_pulses, pulse_rx, channel, pulse_tx.shape[0] // 2)
        awgn_isi_ser_list.append(SER.item())
        print(f"SNR: {snr_db}, AWGN SER: {awgn_ser_list[-1]}, AWGN with ISI SER: {awgn_isi_ser_list[-1]}")

        # Print # run
        print(f"Run: {_ + 1}")

    # Average SERs
    awgn_SERs.append(np.mean(awgn_ser_list))
    awgn_isi_SERs.append(np.mean(awgn_isi_ser_list))

# Plotting
plt.plot(SNRs, theoretical_SERs, label="Theoretical SER")
plt.plot(SNRs, awgn_SERs, label="AWGN Channel SER (Avg)")
plt.plot(SNRs, awgn_isi_SERs, label="AWGN with ISI Channel SER (Avg)")
plt.xlabel("SNR (dB)")
plt.ylabel("SER")
plt.yscale("log")
plt.legend()
plt.grid()
plt.show()
