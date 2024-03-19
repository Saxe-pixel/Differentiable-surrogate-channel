import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam

# Assuming lib.utility and lib.channels are correctly implemented
from lib.utility import estimate_delay, find_closest_symbol
from lib.channels import AWGNChannel, AWGNChannelWithLinearISI

class PulseShapingFilter(nn.Module):
    def __init__(self, filter_length):
        super(PulseShapingFilter, self).__init__()
        # Initialize filter coefficients as trainable parameters
        self.coefficients = nn.Parameter(torch.randn(1, 1, filter_length, dtype=torch.float32))

    def forward(self, x):
        # Apply convolution operation with padding to maintain signal length
        return torch.nn.functional.conv1d(x, self.coefficients, padding='same')

class AWGNChannelSimulator(nn.Module):
    def __init__(self, snr_db):
        super(AWGNChannelSimulator, self).__init__()
        self.snr_db = snr_db

    def forward(self, x):
        snr_linear = 10 ** (self.snr_db / 10.0)
        signal_power = torch.mean(x ** 2)
        noise_variance = signal_power / snr_linear
        noise = torch.randn_like(x) * torch.sqrt(noise_variance)
        return x + noise

def optimize_filter(N_SYMBOLS, SPS, snr_db, pam_symbols, filter_length):
    torch.manual_seed(12345)  # Seed for reproducibility
    
    # Generate random symbols and prepare for pulse shaping
    tx_symbols = torch.tensor(np.random.choice(pam_symbols, N_SYMBOLS), dtype=torch.float32).view(1, 1, -1)
    tx_symbols_upsampled = torch.nn.functional.interpolate(tx_symbols, scale_factor=SPS, mode='nearest')
    
    filter_model = PulseShapingFilter(filter_length)
    channel_model = AWGNChannelSimulator(snr_db)
    optimizer = Adam(filter_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(1000):  # Adjust the number of epochs as needed
        optimizer.zero_grad()

        # Filter and send through AWGN channel
        filtered_signal = filter_model(tx_symbols_upsampled)
        received_signal = channel_model(filtered_signal)
        
        # Apply the same filter as a matched filter in the receiver
        rx_filtered_signal = filter_model(received_signal)
        
        # Downsample the received signal to symbol rate
        rx_symbols_downsampled = rx_filtered_signal[:, :, ::SPS]
        
        # Loss calculation (MSE between transmitted and received symbols)
        loss = criterion(rx_symbols_downsampled, tx_symbols)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:  # Adjust the reporting frequency as needed
            # Calculate SER
            rx_symbols_flattened = rx_symbols_downsampled.flatten()
            closest_symbols = find_closest_symbol(rx_symbols_flattened, torch.tensor(pam_symbols, dtype=torch.float32))
            ser = torch.mean((closest_symbols != tx_symbols.flatten()).float()).item()
            
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, SER: {ser:.4f}")

if __name__ == "__main__":
    N_SYMBOLS = 1000  # Number of symbols
    SPS = 8  # Samples per symbol, for upsampling
    SNR_DB = 10  # Signal-to-noise ratio in dB
    pam_symbols = np.array([-3, -1, 1, 3])  # PAM4 symbols
    FILTER_LENGTH = 101  # Length of the pulse shaping filter
    
    optimize_filter(N_SYMBOLS, SPS, SNR_DB, pam_symbols, FILTER_LENGTH)
