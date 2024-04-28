
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
from lib.channels import AWGNChannel, AWGNChannelWithLinearISI

import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

def forward_pass(tx_symbols, pulse_tx, channel, pulse_rx, padding):
    # Upsample and apply pulse shaping
    tx_symbols_up = torch.zeros((tx_symbols.numel() * SPS,), dtype=torch.double)
    tx_symbols_up[0::SPS] = tx_symbols.double()
    
    # Apply convolution with consistent padding
    x = F.conv1d(tx_symbols_up.view(1, 1, -1), pulse_tx.view(1, 1, -1), padding=padding)

    # Simulate the channel
    y = channel.forward(x.squeeze())

    # Apply receiver with consistent padding
    rx = F.conv1d(
        y.view(1, 1, -1), 
        pulse_rx.view(1, 1, -1).flip(dims=[2]), 
        padding=padding
    ).squeeze()

    return rx, tx_symbols_up, y

if __name__ == "__main__":
    # Simulation parameters
    # (unchanged parameters here)
    SEED = 12345
    N_SYMBOLS = int(1e5)
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
    gg = np.convolve(g, g[::-1])
    pulse_energy = np.max(gg)

    # Define the pulse for the transmitter (to be optimized)
    pulse_tx = torch.from_numpy(g).double().requires_grad_(True)
    # pulse_tx =  torch.from_numpy(g).float()

    # Define the pulse for the receiver (fixed)
    pulse_rx = torch.from_numpy(g).double()  # No requires_grad_() as it's not being optimized

    optimizer = torch.optim.Adam([pulse_tx], lr=0.001)
    channel = AWGNChannelWithLinearISI(snr_db=SNR_DB, pulse_energy=pulse_energy, samples_pr_symbol=SPS)
    # channel = AWGNChannel(snr_db=SNR_DB, pulse_energy=pulse_energy)
    


    # Calculate padding to achieve "same" output length

    sym_trim = FILTER_LENGTH // 2 // SPS
    #####
    num_epochs = 50  # Number of iterations for optimization
    batch_size = 512  # Batch size for optimization
    for epoch in range(num_epochs):
        permutation = torch.randperm(N_SYMBOLS)  
        for i in range(0, N_SYMBOLS, batch_size):
            indices = permutation[i:i+batch_size]
            batch_tx_symbols = tx_symbols[indices]

            optimizer.zero_grad()  # Clear gradients

            # Forward pass (using pulse_tx for transmission)
            tx_symbols_up = torch.zeros((batch_tx_symbols.numel() * SPS, ), dtype=torch.double)
            tx_symbols_up[0::SPS] = batch_tx_symbols.double()

            
            x = F.conv1d(tx_symbols_up.view(1, 1, -1), pulse_tx.view(1, 1, -1), padding=pulse_tx.shape[0]//2)

            
            # Simulate the channel
            y = channel.forward(x.squeeze())

            # Receiver operations (using pulse_rx for reception)
            rx = F.conv1d(y.view(1, 1, -1), pulse_rx.view(1, 1, -1).flip(dims=[2]), padding=pulse_rx.shape[0]//2).squeeze()

            # Compute loss
            
            rx_trimmed = rx[sym_trim:-sym_trim]  # Trim the convolution tails
            tx_symbols_up = tx_symbols_up[sym_trim:-sym_trim]  # Trim the convolution tails
            min_length = min(len(rx_trimmed), len(tx_symbols_up))
            rx_trimmed = rx_trimmed[:min_length]
            tx_symbols_up = tx_symbols_up[:min_length]

            loss = F.mse_loss(rx_trimmed, tx_symbols_up)  # Mean squared error loss

            # Backward pass and update
            loss.backward()
            optimizer.step()


        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    
    
    # required_padding = pulse_tx.shape[0] // 2
    # padded_input = F.pad(tx_symbols_up, (required_padding, required_padding), mode='constant', value=0)
    
    #####
    # Calculate some needed statistics on the pulse - used later for synchronization
    gg = np.convolve(g, g[::-1])
    pulse_energy = np.max(gg)
    print(f"Energy of pulse is: {pulse_energy}")
    sync_point = np.argmax(gg)

    # Transmit - up-sample the symbols and apply pulse
    tx_symbols_up = torch.zeros((N_SYMBOLS * SPS, ), dtype=torch.double)
    tx_symbols_up[0::SPS] = tx_symbols
    
    x = taf.convolve(tx_symbols_up, pulse_tx)  # Apply pulse shaping

    # Apply the "unknown" channel

    
    channel = AWGNChannelWithLinearISI(snr_db=SNR_DB, pulse_energy=pulse_energy, samples_pr_symbol=SPS)
    # channel = AWGNChannel(snr_db=SNR_DB, pulse_energy=pulse_energy)
    y = channel.forward(x)
    # rx, tx_symbols_up, y = forward_pass(tx_symbols_up, pulse_tx, channel, pulse_rx, padding=pulse_tx.shape[0]//2)

    # Apply receiver


    rx = taf.convolve(y, torch.flip(pulse_rx, (0,))) / pulse_energy
    # rx = F.conv1d(y.view(1, 1, -1), pulse_rx.view(1, 1, -1).flip(dims=[2]), padding=pulse_rx.shape[0]//2).squeeze()

    # Correct for the pulse-convolutions
    rx = rx[sync_point::]

    # Decision-making - if channel has introduced a delay we need to correct for that
    delay = estimate_delay(rx, sps=SPS)
    print(f"Delay was estimated to be: {delay}")
    symbols_est = rx[delay::SPS]  # pick out every SPS samples to get the symbols
    symbols_est = symbols_est[0:N_SYMBOLS]  # truncate the tail
    symbols_est = find_closest_symbol(symbols_est, torch.from_numpy(pam_symbols))

    # Count symbol errors
    symbol_error_rate = torch.sum(torch.logical_not(torch.eq(symbols_est, tx_symbols))) / len(tx_symbols)
    print(f"SER: {symbol_error_rate:.3e}")

    # FIXME: Implement the EsN0 dB theory.

    x_detached = x.detach().cpu().numpy().reshape(-1)
    y_detached = y.detach().cpu().numpy().reshape(-1)

    # Plot a subsegment of the transmitted signal and the pulse
    fig, ax = plt.subplots(nrows=2)
    ax[0].plot(t, g)
    ax[0].grid()
    ax[0].set_title('Pulse shaper')

    time_slice = slice(np.argmax(g)-1, np.argmax(g)-1 + 16 * SPS)
    ax[1].plot(x_detached[time_slice], label='Tx signal')
    ax[1].plot(y_detached[time_slice], label='Received signal')
    ax[1].grid()
    ax[1].legend()
    ax[1].set_title('Slice of signals')
    plt.tight_layout()

    # Plot distribution of symbols and plot symbol error rate
    fig, ax = plt.subplots()
    ax.hist(rx.detach().cpu().numpy()[::SPS], bins=100)
    ax.set_title('Distribution of signal after matched filter')

    plt.show()