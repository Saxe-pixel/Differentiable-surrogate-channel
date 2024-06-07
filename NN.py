import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from commpy.filters import rrcosfilter
from lib.utility import estimate_delay, find_closest_symbol
from lib.channels import AWGNChannel, AWGNChannelWithLinearISI, WienerHammersteinISIChannel
import torch.optim as optim
from scipy.special import erfc
import torch
import torch.nn as nn
import torch.optim as optim

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


##############################################

class PulseShapeNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PulseShapeNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    

def train_model(tx_symbols, channel, num_epochs, batch_size):
    # Define neural network parameters
    input_size = 1  # Simplified input to the network
    hidden_size = 100  # Configurable hidden layer size
    output_size = FILTER_LENGTH  # Output the optimized pulse shape
    
    # Initialize the pulse shaping neural network and move it to the appropriate device
    pulse_shape_net = PulseShapeNet(input_size, hidden_size, output_size).double().to(device)
    pulse_shape_net.train()  # Set the network to training mode

    # Define optimizer and loss function
    optimizer = optim.Adam(pulse_shape_net.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        permutation = torch.randperm(train_size)
        for i in range(0, train_size, batch_size):
            indices = permutation[i:i + batch_size]
            batch_tx_symbols = tx_symbols[indices].double().to(device)

            # Neural network generates the pulse
            optimized_pulse = pulse_shape_net(torch.ones(1, 1).double().to(device)).view(1, 1, -1)

            # Forward pass through the communication system
            rx, _ = forward_pass(batch_tx_symbols, optimized_pulse, channel, pulse_rx, FILTER_LENGTH // 2)

            # Loss computation
            rx_trimmed = rx[:batch_tx_symbols.numel()]
            loss = criterion(rx_trimmed, batch_tx_symbols)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    return pulse_shape_net

channel = WienerHammersteinISIChannel(snr_db=6, pulse_energy=pulse_energy, samples_pr_symbol=SPS)
num_epochs = 100
batch_size = 512
trained_net = train_model(train_symbols, channel, num_epochs, batch_size)


def evaluate_with_trained_net(tx_symbols_input, trained_net, receiver_rx, channel, padding):
    trained_net.eval()  # Ensure the network is in evaluation mode
    with torch.no_grad():
        # Generate the optimized pulse shape using the neural network
        optimized_pulse = trained_net(torch.ones(1, 1).double().to(device)).view(1, 1, -1)

        # Prepare the input symbols for evaluation
        tx_symbols_eval = torch.zeros((tx_symbols_input.numel() * SPS,), dtype=torch.double).to(device)
        tx_symbols_eval[0::SPS] = tx_symbols_input.double()

        # Communication system forward pass
        x = F.conv1d(tx_symbols_eval.view(1, 1, -1), optimized_pulse, padding=padding)
        y = channel.forward(x.squeeze())
        rx_eval = F.conv1d(y.view(1, 1, -1), receiver_rx.view(1, 1, -1).flip(dims=[2]), padding=padding).squeeze()

        # Delay estimation and symbol synchronization
        delay = estimate_delay(rx_eval, SPS)
        symbols_est = rx_eval[delay::SPS]
        symbols_est = symbols_est[:test_size]

        # Symbol decision
        symbols_est = find_closest_symbol(symbols_est, torch.from_numpy(pam_symbols))

        # Error calculation
        error = torch.sum(torch.logical_not(torch.eq(symbols_est, tx_symbols_input)))
        SER = error.float() / len(tx_symbols_input)
        return SER

# Example usage within the simulation
SNRs = range(0, 9)  # Range of SNR values to evaluate
wh_isi_SERs = []

for snr_db in SNRs:
    channel = WienerHammersteinISIChannel(snr_db=snr_db, pulse_energy=pulse_energy, samples_pr_symbol=SPS)
    SER = evaluate_with_trained_net(test_symbols, trained_net, pulse_rx, channel, FILTER_LENGTH // 2)
    wh_isi_SERs.append(SER.item())  # Store SER for plotting

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(SNRs, wh_isi_SERs, label="WienerHammerstein ISI Channel SER")
plt.xlabel("SNR (dB)")
plt.ylabel("SER")
plt.yscale("log")
plt.title("SER vs SNR for WienerHammerstein ISI Channel")
plt.legend()
plt.grid(True)
plt.show()
