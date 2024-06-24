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

import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"




# Simulation parameters
# (unchanged parameters here)
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

# split the data into training and testing
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
# zero fitler
pulse_tx = torch.zeros((FILTER_LENGTH,), dtype=torch.double).requires_grad_(True)
# pulse_tx = torch.from_numpy(g).double().requires_grad_(True)
# pulse_tx =  torch.from_numpy(g).float()

# Define the pulse for the receiver (fixed)
pulse_rx = torch.from_numpy(g).double()  # No requires_grad_() as it's not being optimized

optimizer = torch.optim.Adam([pulse_tx], lr=0.001)
channel = AWGNChannelWithLinearISI(snr_db=SNR_DB, pulse_energy=pulse_energy, samples_pr_symbol=SPS)
# channel = AWGNChannel(snr_db=SNR_DB, pulse_energy=pulse_energy)
# channel = WienerHammersteinISIChannel(snr_db=SNR_DB, pulse_energy=pulse_energy, samples_pr_symbol=SPS)



# Calculate padding to achieve "same" output length

sym_trim = FILTER_LENGTH // 2 // SPS
num_epochs = 30  # Number of iterations for optimization
batch_size = 512  # Batch size for optimization
#####
# Training
def forward_pass(tx_symbols_input, optimized_pulse, channel, reciever_rx, padding):
    # Upsample and apply pulse shaping
    tx_symbols_up = torch.zeros((tx_symbols_input.numel() * SPS,), dtype=torch.double)
    tx_symbols_up[0::SPS] = tx_symbols_input.double()
    
    # Apply convolution with consistent padding
    x = F.conv1d(tx_symbols_up.view(1, 1, -1), optimized_pulse.view(1, 1, -1), padding=padding)

    # Simulate the channel
    y = channel.forward(x.squeeze())

    # Apply receiver with consistent padding
    rx = F.conv1d(
        y.view(1, 1, -1), 
        reciever_rx.view(1, 1, -1).flip(dims=[2]), 
        padding=padding
    ).squeeze()
    delay = estimate_delay(rx, SPS)
    rx = rx[delay::SPS]
    rx = rx[:tx_symbols_input.numel()]
    return rx, tx_symbols_up

def train_model(tx_symbols, optimized_pulse, reciever_rx, optimizer, channel, num_epochs, batch_size):
    for epoch in range(num_epochs):
        permutation = torch.randperm(train_size)  
        for i in range(0, train_size, batch_size):
            indices = permutation[i:i+batch_size]
            batch_tx_symbols = tx_symbols[indices].double()

            optimizer.zero_grad()
            rx, _= forward_pass(batch_tx_symbols, optimized_pulse, channel, reciever_rx, optimized_pulse.shape[0]//2)
            
            # Compute loss
            rx_trimmed = rx# [sym_trim:-sym_trim]  # Trim the convolution tails
            batch_tx_symbols_up = batch_tx_symbols # [sym_trim:-sym_trim]  # Trim the convolution tails
            min_length = min(len(rx_trimmed), len(batch_tx_symbols_up))
            rx_trimmed = rx_trimmed[:min_length]
            batch_tx_symbols_up = batch_tx_symbols_up[:min_length]



            loss = F.mse_loss(rx_trimmed, batch_tx_symbols_up)  # Mean squared error loss
            loss.backward()
            optimizer.step()
        # print(f'rx: {rx.shape}, tx_symbols: {batch_tx_symbols.shape}')
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    return optimized_pulse



def evaluate_model(tx_symbols_input, optimized_pulse, reciever_rx, channel, padding):
    with torch.no_grad():
        
        # print(f'tx_symbols_input: {tx_symbols_input.shape}')
        tx_symbols_eval = torch.zeros((tx_symbols_input.numel() * SPS,), dtype=torch.double)
        tx_symbols_eval[0::SPS] = tx_symbols_input.double()
        # print(f'tx_symbols_up: {tx_symbols_eval.shape}')
        # Apply convolution with consistent padding
        x = F.conv1d(tx_symbols_eval.view(1, 1, -1), optimized_pulse.view(1, 1, -1), padding=padding)

        # Simulate the channel
        y = channel.forward(x.squeeze())

        # Apply receiver with consistent padding
        rx_eval = F.conv1d(
            y.view(1, 1, -1), 
            reciever_rx.view(1, 1, -1).flip(dims=[2]), 
            padding=padding
        ).squeeze()
        # print(f'rx: {rx_eval.shape}')

        rx_trimmed = rx_eval# [sym_trim:-sym_trim]
        tx_symbols_trimmed = tx_symbols
        # Call find_closest_symbol 
        delay = estimate_delay(rx_eval, SPS)
        symbols_est = rx_trimmed[delay::SPS]
        symbols_est = symbols_est[:test_size]
        symbols_est = find_closest_symbol(symbols_est, torch.from_numpy(pam_symbols))

        

        # Calculate errors and SER
        # print(symbols_est.shape, tx_symbols_input.shape)
        error = torch.sum(torch.logical_not(torch.eq(symbols_est, tx_symbols_input)))
        SER = error.float() / len(tx_symbols_input)
        return SER



optimized_pulses = train_model(train_symbols, pulse_tx, pulse_rx, optimizer, channel, num_epochs, batch_size) 
SER = evaluate_model(test_symbols, optimized_pulses, pulse_rx, channel, pulse_tx.shape[0]//2)
print(f"SER: {SER}")

    # for epoch in range(num_epochs):
    #     permutation = torch.randperm(N_SYMBOLS)  
    #     for i in range(0, N_SYMBOLS, batch_size):
    #         indices = permutation[i:i+batch_size]
    #         batch_tx_symbols = tx_symbols[indices]

    #         optimizer.zero_grad()  # Clear gradients

    #         # Forward pass (using pulse_tx for transmission)
    #         tx_symbols_up = torch.zeros((batch_tx_symbols.numel() * SPS, ), dtype=torch.double)
    #         tx_symbols_up[0::SPS] = batch_tx_symbols.double()

            
    #         x = F.conv1d(tx_symbols_up.view(1, 1, -1), pulse_tx.view(1, 1, -1), padding=pulse_tx.shape[0]//2)

            
    #         # Simulate the channel
    #         y = channel.forward(x.squeeze())

    #         # Receiver operations (using pulse_rx for reception)
    #         rx = F.conv1d(y.view(1, 1, -1), pulse_rx.view(1, 1, -1).flip(dims=[2]), padding=pulse_rx.shape[0]//2).squeeze()

    #         # Compute loss
            
    #         rx_trimmed = rx[sym_trim:-sym_trim]  # Trim the convolution tails
    #         tx_symbols_up = tx_symbols_up[sym_trim:-sym_trim]  # Trim the convolution tails
    #         min_length = min(len(rx_trimmed), len(tx_symbols_up))
    #         rx_trimmed = rx_trimmed[:min_length]
    #         tx_symbols_up = tx_symbols_up[:min_length]

    #         loss = F.mse_loss(rx_trimmed, tx_symbols_up)  # Mean squared error loss

    #         # Backward pass and update
    #         loss.backward()
    #         optimizer.step()


    #     print(f"Epoch {epoch+1}, Loss: {loss.item()}")