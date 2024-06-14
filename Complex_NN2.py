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
pulse_rx = torch.from_numpy(g).double().to(device)  # Receiver pulse (fixed)
pulse_energy = np.max(gg)

class WHChannelNet(nn.Module):
    def __init__(self, filter_length, num_filters=1):
        super(WHChannelNet, self).__init__()
        self.conv1 = nn.Conv1d(1, num_filters, filter_length, padding=filter_length // 2, bias=False, dtype=torch.double)
        self.conv2 = nn.Conv1d(num_filters, num_filters, filter_length, padding=filter_length // 2, bias=False, dtype=torch.double)
        self.conv3 = nn.Conv1d(num_filters, num_filters, filter_length, padding=filter_length // 2, bias=False, dtype=torch.double)
        self.conv4 = nn.Conv1d(num_filters, 1, filter_length, padding=filter_length // 2, bias=False, dtype=torch.double)
        
        # Adding batch normalization layers
        self.bn1 = nn.BatchNorm1d(num_filters, dtype=torch.double)
        self.bn2 = nn.BatchNorm1d(num_filters, dtype=torch.double)
        self.bn3 = nn.BatchNorm1d(num_filters, dtype=torch.double)

        # Adding residual connections
        self.residual1 = nn.Conv1d(1, num_filters, 1, bias=False, dtype=torch.double)
        self.residual2 = nn.Conv1d(num_filters, num_filters, 1, bias=False, dtype=torch.double)
        
        # Adding fully connected layers for non-linear transformation
        self.fc1 = nn.Linear(num_filters, num_filters, dtype=torch.double)
        self.fc2 = nn.Linear(num_filters, num_filters, dtype=torch.double)
        self.fc3 = nn.Linear(num_filters, num_filters, dtype=torch.double)

        # Initializing the weights of the layers
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.xavier_uniform_(self.residual1.weight)
        nn.init.xavier_uniform_(self.residual2.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        
        res1 = self.residual1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = x[:, :, :res1.shape[2]]  # Ensure the size matches for residual connection
        x = x + res1  # First residual connection
        #print(f"Shape after first residual: {x.shape}")
        
        res2 = self.residual2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = x[:, :, :res2.shape[2]]  # Ensure the size matches for residual connection
        x = x + res2  # Second residual connection
        #print(f"Shape after second residual: {x.shape}")
        
        batch_size, num_filters, seq_length = x.size()
        x = x.permute(0, 2, 1).contiguous().view(-1, num_filters)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x = x.view(batch_size, seq_length, num_filters).permute(0, 2, 1).contiguous()

        x = self.conv4(x)
        #print(f"Output shape: {x.shape}")
        return x


# class WHChannelNet(nn.Module):
#     def __init__(self, filter_length, num_filters=64, initial_non_linear_coefficients=(0.0, 0.0, 0.0)):
#         super(WHChannelNet, self).__init__()
#         self.conv1 = nn.Conv1d(1, num_filters, filter_length, padding=filter_length // 2, bias=False, dtype=torch.double)
#         self.conv2 = nn.Conv1d(num_filters, num_filters, filter_length, padding=filter_length // 2, bias=False, dtype=torch.double)
#         self.conv3 = nn.Conv1d(num_filters, 1, filter_length, padding=filter_length // 2, bias=False, dtype=torch.double)
        
#         # Adding batch normalization layers
#         self.bn1 = nn.BatchNorm1d(num_filters, dtype=torch.double)
#         self.bn2 = nn.BatchNorm1d(num_filters, dtype=torch.double)
        
#         # Adding residual connections
#         self.residual = nn.Conv1d(1, num_filters, 1, bias=False, dtype=torch.double)
        
#         # Initialize the non-linear coefficients as learnable parameters
#         self.a0 = nn.Parameter(torch.tensor(initial_non_linear_coefficients[0], dtype=torch.double))
#         self.a1 = nn.Parameter(torch.tensor(initial_non_linear_coefficients[1], dtype=torch.double))
#         self.a2 = nn.Parameter(torch.tensor(initial_non_linear_coefficients[2], dtype=torch.double))

#         # Initialize the weights of the convolutional layers
#         nn.init.xavier_uniform_(self.conv1.weight)
#         nn.init.xavier_uniform_(self.conv2.weight)
#         nn.init.xavier_uniform_(self.conv3.weight)
#         nn.init.xavier_uniform_(self.residual.weight)

#     def forward(self, x):
#         res = self.residual(x)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = x[:, :, :res.shape[2]]  # Ensure the size matches for residual connection
#         x = x + res  # Residual connection
#         x = self.a0 * x + self.a1 * x ** 2 + self.a2 * x ** 3
#         x = self.conv3(x)
#         return x

# Training Function
def train_pulse_shaping_net(tx_symbols, network, receiver_rx, optimizer, channel, num_epochs, batch_size, sps):
    for epoch in range(num_epochs):
        permutation = torch.randperm(tx_symbols.size(0))  # Shuffle the indices
        total_loss = 0
        for i in range(0, tx_symbols.size(0), batch_size):
            indices = permutation[i:i + batch_size]  # Get batch indices
            batch_tx_symbols = tx_symbols[indices].double().to(device)  # Extract batch

            optimizer.zero_grad()  # Reset gradients

            # Upsample input symbols
            tx_symbols_up = torch.zeros((batch_tx_symbols.numel() * sps,), dtype=torch.double).to(device)
            tx_symbols_up[0::sps] = batch_tx_symbols

            # Forward pass through the network (pulse shaping)
            shaped_pulse = network(tx_symbols_up.view(1, 1, -1))

            # Debug: Check shapes
            #print(f"Shaped pulse shape: {shaped_pulse.shape}")

            # Pass through the actual WH channel
            y = channel.forward(shaped_pulse.squeeze())

            # Debug: Check shapes
            #print(f"Channel output shape: {y.shape}")

            # Pass through the receiver filter
            rx = torch.nn.functional.conv1d(y.view(1, 1, -1), receiver_rx.view(1, 1, -1).flip(dims=[2]), padding=(receiver_rx.shape[0] - 1) // 2).squeeze()

            # Debug: Check shapes
            #print(f"Receiver output shape: {rx.shape}")

            # Delay estimation and synchronization
            delay = estimate_delay(rx, sps)
            rx = rx[delay::sps]
            rx = rx[:batch_tx_symbols.numel()]

            # Debug: Check shapes
            #print(f"Synchronized received shape: {rx.shape}")

            # Loss calculation and backpropagation
            loss = torch.nn.functional.mse_loss(rx, batch_tx_symbols)
            loss.backward()  # Compute gradients
            optimizer.step()  # Update parameters

            total_loss += loss.item()  # Accumulate loss

        # Print average loss for the epoch
        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / (i // batch_size + 1)}")
    return network


# Evaluation Function
def evaluate_model(tx_symbols_input, network, receiver_rx, channel, sps):
    with torch.no_grad():
        # Prepare input signal
        tx_symbols_up = torch.zeros((tx_symbols_input.numel() * sps,), dtype=torch.double).to(device)
        tx_symbols_up[0::sps] = tx_symbols_input.double()

        # Forward pass through the network and actual WH channel
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

# Theoretical SER Calculation
def theoretical_ser(snr_db, pulse_energy, modulation_order):
    log2M = np.log2(modulation_order)
    SNR_linear = 10 ** (snr_db / 10)
    Eb_N0_linear = SNR_linear / log2M
    Q = lambda x: 0.5 * erfc(x / np.sqrt(2))
    SER_theoretical = 2 * (1 - 1 / modulation_order) * Q(np.sqrt(2 * log2M * Eb_N0_linear))
    return SER_theoretical

# Initialize pulse shaping network and optimizer
def create_network_and_optimizer():
    network = WHChannelNet(FILTER_LENGTH).to(device)
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    return network, optimizer

# SNR settings and results storage
SNRs = range(0, 2)
num_epochs = 3
batch_size = 512

theoretical_SERs = [theoretical_ser(snr_db, pulse_energy, 4) for snr_db in SNRs]
wh_isi_SERs = []

# Simulation loop
for snr_db in SNRs:
    print(f"Training and evaluating at SNR: {snr_db} dB")
    wiener_hammerstein_channel = WienerHammersteinISIChannel(snr_db=snr_db, pulse_energy=pulse_energy, samples_pr_symbol=SPS)
    
    # Re-initialize network and optimizer for each SNR
    complex_wh_net, optimizer = create_network_and_optimizer()
    
    # Train the pulse shaping network using the actual WH channel
    trained_network_wh = train_pulse_shaping_net(train_symbols, complex_wh_net, pulse_rx, optimizer, wiener_hammerstein_channel, num_epochs, batch_size, SPS)
    ser_wh = evaluate_model(test_symbols, trained_network_wh, pulse_rx, wiener_hammerstein_channel, SPS)
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
