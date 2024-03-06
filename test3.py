import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
import matplotlib.pyplot as plt  # Import matplotlib for plotting (optional)

from lib.utility import estimate_delay, find_closest_symbol
from lib.channels import AWGNChannel, AWGNChannelWithLinearISI


class PulseShapingFilter(nn.Module):
    def __init__(self, filter_length):
        super(PulseShapingFilter, self).__init__()
        # Initialize the filter coefficients as a parameter
        self.coefficients = nn.Parameter(torch.rand(1, 1, filter_length, dtype=torch.float32))

    def forward(self, x):
        # Apply the convolution operation
        return torch.nn.functional.conv1d(x, self.coefficients, padding='same')

class AWGNChannelSimulator(nn.Module):
    def __init__(self, snr_db):
        super(AWGNChannelSimulator, self).__init__()
        self.snr_db = snr_db

    def forward(self, x):
        snr = 10 ** (self.snr_db / 10.0)
        signal_power = torch.mean(x ** 2)
        noise_variance = signal_power / snr
        noise = torch.randn_like(x) * torch.sqrt(noise_variance)
        return x + noise

def optimize_filter(N_SYMBOLS, SPS, snr_db, pam_symbols, filter_length):
    torch.manual_seed(12345)  # Ensure reproducibility
    
    tx_symbols = torch.tensor(np.random.choice(pam_symbols, size=N_SYMBOLS), dtype=torch.float32).view(-1, 1)
    pam_symbols_tensor = torch.tensor(pam_symbols, dtype=torch.float32)  # Convert PAM symbols to tensor
    filter_model = PulseShapingFilter(filter_length)
    channel_model = AWGNChannelSimulator(snr_db=snr_db)
    optimizer = Adam(filter_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print_interval = 1000  # Print loss and SER every 1000 epochs

    for epoch in range(10000):  # Run optimization for 10000 epochs
        optimizer.zero_grad()

        tx_symbols_up = torch.zeros(N_SYMBOLS * SPS, 1, dtype=torch.float32)
        tx_symbols_up[::SPS, 0] = tx_symbols.squeeze()  # Upsampling

        tx_symbols_up = tx_symbols_up.transpose(0, 1).unsqueeze(0)  # Reshape for conv1d

        filtered_signal = filter_model(tx_symbols_up)
        received_signal = channel_model(filtered_signal)

        rx_symbols = filter_model(received_signal).squeeze(0).transpose(0, 1)
        delay = estimate_delay(rx_symbols, SPS)  # Estimate delay
        rx_symbols_adjusted = rx_symbols[delay:, :]  # Adjust for delay
        rx_downsampled = rx_symbols_adjusted[::SPS, :][:N_SYMBOLS]  # Downsampling and adjusting for batch size

        loss = criterion(rx_downsampled, tx_symbols)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % print_interval == 0:
            # Calculate and print SER including delay correction
            closest_symbols = find_closest_symbol(rx_downsampled, pam_symbols_tensor)
            ser = torch.mean((closest_symbols != tx_symbols).float())
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, SER: {ser.item():.4f}")

    return rx_downsampled, tx_symbols  # Return for later processing (optional)

if __name__ == "__main__":
    N_SYMBOLS = 1000
    SPS = 8
    SNR_DB = 10
    pam_symbols = np.array([-3, -1, 1, 3])
    FILTER_LENGTH = 101

    rx_downsampled, tx_symbols = optimize_filter(N_SYMBOLS, SPS, SNR_DB, pam_symbols, FILTER_LENGTH)

    

