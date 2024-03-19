import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam

class PulseShapingFilter(nn.Module):
    def __init__(self, filter_length):
        super(PulseShapingFilter, self).__init__()
        self.coefficients = nn.Parameter(torch.randn(1, 1, filter_length, dtype=torch.float32))

    def forward(self, x):
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

# Custom loss function focusing on minimizing the distance to the nearest PAM symbol
def custom_loss(output, target, pam_symbols_tensor):
    # Calculate distances to each PAM symbol and find the minimum distance
    distances = torch.abs(output.unsqueeze(-1) - pam_symbols_tensor)  # Expand output for broadcasting
    min_distances, _ = torch.min(distances, dim=2)
    return torch.mean(min_distances)

def optimize_filter(N_SYMBOLS, SPS, snr_db, pam_symbols, filter_length):
    torch.manual_seed(12345)
    
    tx_symbols = torch.tensor(np.random.choice(pam_symbols, N_SYMBOLS), dtype=torch.float32).view(1, 1, -1)
    tx_symbols_upsampled = torch.nn.functional.interpolate(tx_symbols, scale_factor=SPS, mode='nearest')
    
    pam_symbols_tensor = torch.tensor(pam_symbols, dtype=torch.float32).view(1, 1, -1)
    filter_model = PulseShapingFilter(filter_length)
    channel_model = AWGNChannelSimulator(snr_db)
    optimizer = Adam(filter_model.parameters(), lr=0.001)

    for epoch in range(1000):  # Adjust the number of epochs as needed
        optimizer.zero_grad()

        filtered_signal = filter_model(tx_symbols_upsampled)
        received_signal = channel_model(filtered_signal)
        rx_filtered_signal = filter_model(received_signal)
        rx_symbols_downsampled = rx_filtered_signal[:, :, ::SPS]

        loss = custom_loss(rx_symbols_downsampled, tx_symbols, pam_symbols_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Custom Loss: {loss.item():.4f}")

if __name__ == "__main__":
    N_SYMBOLS = 1000
    SPS = 8
    SNR_DB = 10
    pam_symbols = np.array([-3, -1, 1, 3])
    FILTER_LENGTH = 101
    
    optimize_filter(N_SYMBOLS, SPS, SNR_DB, pam_symbols, FILTER_LENGTH)
