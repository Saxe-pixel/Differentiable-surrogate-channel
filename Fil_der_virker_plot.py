
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchaudio.functional as taf
from commpy.filters import rrcosfilter
import torch.nn.functional as F
from lib.utility import estimate_delay, find_closest_symbol
from lib.channels import AWGNChannel, AWGNChannelWithLinearISI, WienerHammersteinISIChannel

# Simulation parameters
SEED = 12345
N_SYMBOLS = int(2*1e5)
SPS = 8  # samples-per-symbol (oversampling rate)
SNR_DB = 4.0  # signal-to-noise ratio in dB
BAUD_RATE = 10e6  # number of symbols transmitted pr. second
FILTER_LENGTH = 64
Ts = 1 / (BAUD_RATE)  # symbol length
fs = BAUD_RATE * SPS
random_obj = np.random.default_rng(SEED)

# Generate data - use Pulse Amplitude Modulation (PAM)
pam_symbols = np.array([-3, -1, 1, 3])
tx_symbols = torch.from_numpy(random_obj.choice(pam_symbols, size=(N_SYMBOLS,), replace=True))

# Construct pulse shape for both transmitter and receiver
t, g = rrcosfilter(FILTER_LENGTH, 0.5, Ts, fs)
g /= np.linalg.norm(g)  # Normalize pulse to have unit energy

# Define the pulse for the transmitter (to be optimized)
pulse_tx = torch.zeros((FILTER_LENGTH,), dtype=torch.double).requires_grad_(True)

# Define the pulse for the receiver (fixed)
pulse_rx = torch.from_numpy(g).double()  # No requires_grad_() as it's not being optimized

# Plot the signals before and after RRC pulse shaping
def plot_signals(tx_symbols, pulse_tx, pulse_rx, sps, num_symbols_to_plot=100):
    # Select a subset of symbols to plot
    tx_symbols_subset = tx_symbols[:num_symbols_to_plot]
    
    # Upsample the symbols
    tx_symbols_up = torch.zeros((tx_symbols_subset.numel() * sps,), dtype=torch.double)
    tx_symbols_up[0::sps] = tx_symbols_subset.double()
    
    # Apply pulse shaping using convolution
    tx_shaped = F.conv1d(tx_symbols_up.view(1, 1, -1), pulse_rx.view(1, 1, -1), padding=pulse_rx.shape[0]//2).squeeze().detach().numpy()
    
    # Plot the original upsampled symbols
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(tx_symbols_up.detach().numpy(), label="Upsampled Symbols")
    plt.title("Upsampled Symbols")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    
    # Plot the pulse shaped signal
    plt.subplot(2, 1, 2)
    plt.plot(tx_shaped, label="Pulse Shaped Signal")
    plt.title("Pulse Shaped Signal with RRC Filter")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Call the plotting function
plot_signals(tx_symbols, pulse_tx, pulse_rx, SPS, num_symbols_to_plot=100)
