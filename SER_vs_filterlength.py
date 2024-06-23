import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from commpy.filters import rrcosfilter
from lib.utility import estimate_delay, find_closest_symbol
from lib.channels import WienerHammersteinISIChannel
from scipy.special import erfc

# Define device and simulation parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 12345
N_SYMBOLS = int(2 * 1e5)
SPS = 8  # Samples-per-symbol (oversampling rate)
BAUD_RATE = 10e6  # Number of symbols transmitted per second
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

class WHChannelNet(nn.Module):
    def __init__(self, filter_length, num_filters, initial_non_linear_coefficients=(0,0,0)):
        super(WHChannelNet, self).__init__()
        self.conv1 = nn.Conv1d(1, num_filters, filter_length, padding=filter_length // 2, bias=False, dtype=torch.double)
        self.conv2 = nn.Conv1d(num_filters, 1, filter_length, padding=filter_length // 2, bias=False, dtype=torch.double)
        
        # Making non-linear coefficients learnable parameters
        self.a0 = nn.Parameter(torch.tensor(initial_non_linear_coefficients[0], dtype=torch.double))
        self.a1 = nn.Parameter(torch.tensor(initial_non_linear_coefficients[1], dtype=torch.double))
        self.a2 = nn.Parameter(torch.tensor(initial_non_linear_coefficients[2], dtype=torch.double))

        # Initializing the weights of the convolutional layers randomly
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.a0 * x + self.a1 * x ** 2 + self.a2 * x ** 3
        x = self.conv2(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Training Function
def train_pulse_shaping_net(tx_symbols, network, receiver_rx, optimizer, channel, num_epochs, batch_size, sps):
    for epoch in range(num_epochs):
        permutation = torch.randperm(tx_symbols.size(0))
        total_loss = 0
        for i in range(0, tx_symbols.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_tx_symbols = tx_symbols[indices].double().to(device)

            optimizer.zero_grad()

            # Upsample input symbols
            tx_symbols_up = torch.zeros((batch_tx_symbols.numel() * sps,), dtype=torch.double).to(device)
            tx_symbols_up[0::sps] = batch_tx_symbols

            # Forward pass through the network (pulse shaping)
            shaped_pulse = network(tx_symbols_up.view(1, 1, -1))

            # Pass through the actual WH channel
            y = channel.forward(shaped_pulse.squeeze())

            # Pass through the receiver filter
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
        print(f"SER: {SER.item()}")
        return SER

# Initialize SNR and results storage
snr_db = 20
num_epochs = 4
batch_size = 512

num_filters = 32
num_runs = 4

results = []

# Simulation loop
for filter_length in range(1, 129, 8):  # Training every 8 filter lengths
    ser_avg = 0.0
    print(f"Training and evaluating with filter_length={filter_length}, num_filters={num_filters}")
    
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        # Initialize transmitter pulse shape as zero filter
        pulse_tx = torch.zeros((filter_length,), dtype=torch.double).to(device)
        t, g = rrcosfilter(filter_length, 0.5, Ts, fs)
        g /= np.linalg.norm(g)  # Normalize pulse to have unit energy
        pulse_rx = torch.from_numpy(g).double().to(device)  # Receiver pulse (fixed)
        pulse_energy = np.max(np.convolve(g, g[::-1]))

        inverse_wh_net = WHChannelNet(filter_length, num_filters).to(device)
        optimizer = optim.Adam(inverse_wh_net.parameters(), lr=0.001)
        wiener_hammerstein_channel = WienerHammersteinISIChannel(snr_db=snr_db, pulse_energy=pulse_energy, samples_pr_symbol=SPS)
        
        # Train the pulse shaping network using the actual WH channel
        trained_network_wh = train_pulse_shaping_net(train_symbols, inverse_wh_net, pulse_rx, optimizer, wiener_hammerstein_channel, num_epochs, batch_size, SPS)
        ser_wh = evaluate_model(test_symbols, trained_network_wh, pulse_rx, wiener_hammerstein_channel, SPS)
        
        ser_avg += ser_wh.item()
    
    ser_avg /= num_runs
    num_parameters = count_parameters(trained_network_wh)
    results.append((filter_length, num_filters, ser_avg, num_parameters))

# Prepare data for plotting
filter_lengths_params = []
ser_values = []

for result in results:
    filter_lengths_params.append(result[3])  # Number of parameters
    ser_values.append(result[2])  # SER

# Plotting results
plt.figure(figsize=(12, 8))
plt.plot(filter_lengths_params, ser_values, marker='o', linestyle='-')
for i, txt in enumerate(results):
    plt.annotate(f"FL={txt[0]}", (filter_lengths_params[i], ser_values[i]))

plt.xlabel("Number of Parameters")
plt.ylabel("Symbol Error Rate (SER)")
plt.yscale("log")
plt.title("SER vs. Number of Parameters for Different Filter Lengths (Averaged over 5 runs)")
plt.grid(True)
plt.show()