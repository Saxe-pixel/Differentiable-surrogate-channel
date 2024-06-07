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
tx_symbols = torch.from_numpy(random_obj.choice(pam_symbols, size=(N_SYMBOLS,), replace=True)).to(device)

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

# Neural Network for Pulse Shaping
# class PulseShapeNet(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(PulseShapeNet, self).__init__()
#         # Assuming input size corresponds to the number of input channels for Conv1d which should be 1
#         # and output size corresponds to the output length of the final layer
#         self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)  # 16 output channels
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)  # 32 output channels
#         self.fc1 = nn.Linear(32 * FILTER_LENGTH, hidden_size)  # Hidden size fully connected
#         self.fc2 = nn.Linear(hidden_size, output_size)  # Output size fully connected

#     def forward(self, x):
#         x = x.view(1, 1, -1)  # Reshape to (batch, channels, length)
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = x.view(1, -1)  # Flatten
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

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
    input_size = 1
    hidden_size = 100
    output_size = FILTER_LENGTH
    pulse_shape_net = PulseShapeNet(input_size, hidden_size, output_size).double().to(device)
    optimizer = optim.Adam(pulse_shape_net.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        permutation = torch.randperm(train_size)
        for i in range(0, train_size, batch_size):
            indices = permutation[i:i + batch_size]
            batch_tx_symbols = tx_symbols[indices].double()

            optimized_pulse = pulse_shape_net(torch.ones((1, 1)).double().to(device)).view(1, 1, -1)

            tx_symbols_up = torch.zeros((batch_tx_symbols.numel() * SPS,), dtype=torch.double).to(device)
            tx_symbols_up[0::SPS] = batch_tx_symbols.double()
            x = F.conv1d(tx_symbols_up.view(1, 1, -1), optimized_pulse, padding=FILTER_LENGTH // 2)
            y = channel.forward(x.squeeze())
            rx = F.conv1d(y.view(1, 1, -1), pulse_rx.view(1, 1, -1).flip(dims=[2]), padding=FILTER_LENGTH // 2).squeeze()

            delay = estimate_delay(rx, SPS)
            rx_trimmed = rx[delay::SPS][:batch_tx_symbols.numel()]

            loss = F.mse_loss(rx_trimmed, batch_tx_symbols)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    return pulse_shape_net

def evaluate_model(tx_symbols_input, trained_net, receiver_rx, channel, padding):
    trained_net.eval()  # Ensure the network is in evaluation mode
    with torch.no_grad():
        # Generate the optimized pulse shape using the neural network
        optimized_pulse = trained_net(torch.ones((1, 1)).double().to(device)).view(1, 1, -1)

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
        symbols_est = symbols_est[:tx_symbols_input.numel()]

        # Symbol decision
        symbols_est = find_closest_symbol(symbols_est, torch.from_numpy(pam_symbols))

        # Error calculation
        error = torch.sum(torch.logical_not(torch.eq(symbols_est, tx_symbols_input)))
        SER = error.float() / len(tx_symbols_input)
        return SER

def theoretical_ser(snr_db, pulse_energy, modulation_order):
    log2M = np.log2(modulation_order)
    SNR_linear = 10 ** (snr_db / 10)
    Eb_N0_linear = SNR_linear / log2M
    Q = lambda x: 0.5 * erfc(x / np.sqrt(2))
    SER_theoretical = 2 * (1 - 1 / modulation_order) * Q(np.sqrt(2 * log2M * Eb_N0_linear))
    return SER_theoretical

# Example usage within the simulation
SNRs = range(0, 9)
theoretical_SERs = [theoretical_ser(snr_db, pulse_energy, 4) for snr_db in SNRs]
awgn_SERs = []
awgn_isi_SERs = []
wh_isi_SERs = []

for snr_db in SNRs:
    # Train and evaluate on AWGNChannel
    awgn_channel = AWGNChannel(snr_db=snr_db, pulse_energy=pulse_energy)
    trained_net_awgn = train_model(train_symbols, awgn_channel, 10, 1024)
    SER_awgn = evaluate_model(test_symbols, trained_net_awgn, pulse_rx, awgn_channel, FILTER_LENGTH // 2)
    awgn_SERs.append(SER_awgn)

    # Train and evaluate on AWGNChannelWithLinearISI
    awgn_isi_channel = AWGNChannelWithLinearISI(snr_db=snr_db, pulse_energy=pulse_energy, samples_pr_symbol=SPS)
    trained_net_awgn_isi = train_model(train_symbols, awgn_isi_channel, 10, 1024)
    SER_awgn_isi = evaluate_model(test_symbols, trained_net_awgn_isi, pulse_rx, awgn_isi_channel, FILTER_LENGTH // 2)
    awgn_isi_SERs.append(SER_awgn_isi)

    # Train and evaluate on WienerHammersteinISIChannel
    wh_channel = WienerHammersteinISIChannel(snr_db=snr_db, pulse_energy=pulse_energy, samples_pr_symbol=SPS)
    trained_net_wh = train_model(train_symbols, wh_channel, 10, 1024)
    SER_wh = evaluate_model(test_symbols, trained_net_wh, pulse_rx, wh_channel, FILTER_LENGTH // 2)
    wh_isi_SERs.append(SER_wh)

# Plotting
plt.plot(SNRs, theoretical_SERs, label="Theoretical SER")
plt.plot(SNRs, awgn_SERs, label="AWGN Channel SER")
plt.plot(SNRs, awgn_isi_SERs, label="AWGN with ISI Channel SER")
plt.plot(SNRs, wh_isi_SERs, label="WienerHammerstein ISI Channel SER")
plt.xlabel("SNR (dB)")
plt.ylabel("SER")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.show()