"""
    Adjusted simulation with neural network for symbol recovery
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from commpy.filters import rrcosfilter
from lib.utility import find_closest_symbol
from lib.channels import AWGNChannel

class SymbolRecoveryCNN(nn.Module):
    def __init__(self, sps):
        super(SymbolRecoveryCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(16, 1, kernel_size=5, stride=sps, padding=2)  # Stride equals samples per symbol, reduces output rate to symbol rate

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x

# Simulation parameters
SEED = 12345
N_SYMBOLS = int(1e5)
SPS = 8  # Samples-per-symbol (oversampling rate)
SNR_DB = 4.0  # Signal-to-noise ratio in dB
BAUD_RATE = 10e6  # Symbols per second
FILTER_LENGTH = 256
Ts = 1 / BAUD_RATE  # Symbol period
fs = BAUD_RATE * SPS  # Sampling frequency

# Random generator for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)

# Generate PAM symbols
pam_symbols = np.array([-3, -1, 1, 3])
tx_symbols = torch.from_numpy(np.random.choice(pam_symbols, size=(N_SYMBOLS,), replace=True)).float()

# Pulse shaping filter
t, g = rrcosfilter(FILTER_LENGTH, 0.5, Ts, fs)
g /= np.linalg.norm(g)  # Normalize
pulse = torch.from_numpy(g).float()

# Upsample symbols
tx_symbols_up = torch.zeros((N_SYMBOLS * SPS,), dtype=torch.float)
tx_symbols_up[0::SPS] = tx_symbols

# Convolve with pulse shape
x = F.conv1d(tx_symbols_up.view(1, 1, -1), pulse.view(1, 1, -1), padding=(pulse.shape[0] // 2))

# Channel
pulse_energy = np.sum(g**2)
channel = AWGNChannel(SNR_DB, pulse_energy)
y = channel.forward(x)

# Model, loss, optimizer
model = SymbolRecoveryCNN(SPS)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
y_reshaped = y.view(1, 1, -1)  # Reshape for Conv1D
for epoch in range(100):  # Number of epochs
    optimizer.zero_grad()
    outputs = model(y_reshaped)
    loss = criterion(outputs.view(-1)[:N_SYMBOLS], tx_symbols)  # Align output and target dimensions
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Post-processing for SER calculation
predicted_symbols = find_closest_symbol(outputs.view(-1)[:N_SYMBOLS].detach(), torch.from_numpy(pam_symbols).float())

# SER Calculation
symbol_error_rate = (predicted_symbols != tx_symbols).float().mean().item()
print(f'Symbol Error Rate: {symbol_error_rate}')


# Plot the pulse shape
fig, ax = plt.subplots(nrows=2)
ax[0].plot(t, g)
ax[0].grid()
ax[0].set_title('Pulse shaper')

# Convert x and y to 1D numpy arrays for plotting
x_numpy = x.detach().cpu().numpy().squeeze()
y_numpy = y.detach().cpu().numpy().squeeze()

# Ensure the time_slice is valid for x and y
# Here, you might need to adjust the slicing based on the length of x and y
# For demonstration, let's use a simple slice for the first 100 samples
# Adjust this as necessary for your specific needs
time_slice = slice(0, 100)  # Example slice; replace with your actual intended slice

ax[1].plot(x_numpy[time_slice], label='Tx signal')
ax[1].plot(y_numpy[time_slice], label='Received signal')
ax[1].grid()
ax[1].legend()
ax[1].set_title('Slice of signals')
plt.tight_layout()

# Assuming 'outputs' are the soft decisions from the CNN
soft_decisions = outputs.view(-1)[:N_SYMBOLS].detach().cpu().numpy()

# Plot the distribution of soft decisions
fig, ax = plt.subplots()
ax.hist(soft_decisions, bins=100, density=True)  # Use density=True for probability distribution
ax.set_title('Distribution of soft decisions after CNN')
plt.show()