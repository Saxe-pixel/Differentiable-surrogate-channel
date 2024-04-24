import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchaudio.functional as taf
from commpy.filters import rrcosfilter
from scipy.special import erfc
import numpy as np

from lib.utility import estimate_delay, find_closest_symbol
from lib.channels import AWGNChannel, AWGNChannelWithLinearISI


# Define a simple CNN class for pulse filter optimization
class PulseFilterCNN(nn.Module):
    def __init__(self, filter_length):
        super(PulseFilterCNN, self).__init__()
        # Define the convolutional layer with appropriate padding
        self.conv1 = nn.Conv1d(1, 1, filter_length, padding=filter_length // 2)
        # Use ReLU activation for non-linearity
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        # Normalize the output to ensure unit energy
        return x / torch.norm(x)


def main():
    # Simulation parameters
    SEED = 12345
    N_SYMBOLS = int(1e5)
    SPS = 8  # samples-per-symbol (oversampling rate)
    SNR_DB = 4.0  # signal-to-noise ratio in dB
    BAUD_RATE = 10e6  # number of symbols transmitted pr. second
    FILTER_LENGTH = 256
    Ts = 1 / (BAUD_RATE)  # symbol length
    fs = BAUD_RATE * SPS

    # Random generator
    random_obj = np.random.default_rng(SEED)

    # Generate data - use a Pulse Amplitude Modulation (PAM)
    pam_symbols = np.array([-3, -1, 1, 3])
    tx_symbols = torch.from_numpy(random_obj.choice(pam_symbols, size=(N_SYMBOLS,), replace=True))

    # Construct pulse shape (initial guess)
    t, g = rrcosfilter(FILTER_LENGTH, 0.5, Ts, fs)
    g /= np.linalg.norm(g)  # ensure that pulse as unit norm
    pulse = torch.from_numpy(g)

    # Calculate some needed statistics on the pulse - used later for synchronization
    gg = np.convolve(g, g[::-1])
    pulse_energy = np.max(gg)
    print(f"Energy of pulse is: {pulse_energy}")
    sync_point = np.argmax(gg)

    # Define the CNN for pulse filter optimization
    cnn_model = PulseFilterCNN(FILTER_LENGTH)

    # Define the loss function (Mean Squared Error between received and original symbols)
    criterion = nn.MSELoss()

    # Define an optimizer (here, Adam optimizer)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(100):  # Adjust number of epochs for better performance
        # Generate random data
        tx_symbols = torch.from_numpy(random_obj.choice(pam_symbols, size=(N_SYMBOLS,), replace=True))

        # Reshape pulse to have a single channel dimension
        pulse_reshaped = pulse.unsqueeze(dim=0)  # Add dimension for batch size (set to 1)

        # Convert pulse to float
        pulse_float = pulse_reshaped.float()

        # Pass the converted pulse to the CNN
        optimized_pulse = cnn_model(pulse_float)

        # Apply the optimized pulse for transmission
        tx_symbols_up = torch.zeros((N_SYMBOLS * SPS,), dtype=torch.double)
        tx_symbols_up[0::SPS] = tx_symbols
        # Reshape tx_symbols_up to add channel dimension
        tx_symbols_up = tx_symbols_up.unsqueeze(dim=1)  
        tx_symbols_up = tx_symbols_up.float()
        x = taf.convolve(tx_symbols_up, optimized_pulse)
        # Apply the "unknown" channel
        # Channel selection (uncomment the desired channel)
        # channel = AWGNChannel(snr_db=SNR_DB, pulse_energy=pulse_energy)
        channel = AWGNChannelWithLinearISI(snr_db=SNR_DB, pulse_energy=pulse_energy, samples_pr_symbol=SPS)
        x = x.squeeze(dim=1)  # Remove the second dimension (likely from convolution)
        y = channel.forward(x)

        # Apply receiver
        rx = taf.convolve(y, torch.flip(optimized_pulse, (0,))) / pulse_energy

        # Correct for pulse-convolutions and delay
        rx = rx[sync_point::]
        delay = estimate_delay(rx, sps=SPS)
        symbols_est = rx[delay::SPS][0:N_SYMBOLS]
        symbols_est = find_closest_symbol(symbols_est, torch.from_numpy(pam_symbols))

        # Calculate the loss
        loss = criterion(symbols_est, tx_symbols)

        # Backpropagation and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training progress (optional)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

    # Evaluate the performance with the optimized pulse

    # Apply the optimized pulse for transmission (evaluation)
    tx_symbols_up = torch.zeros((N_SYMBOLS * SPS,), dtype=torch.double)
    tx_symbols_up[0::SPS] = tx_symbols
    x_eval = taf.convolve(tx_symbols_up, optimized_pulse.detach())

    # Apply the channel and receiver (evaluation)
    y_eval = channel.forward(x_eval)
    rx_eval = taf.convolve(y_eval, torch.flip(optimized_pulse.detach(), (0,))) / pulse_energy

    # Correct for pulse-convolutions and delay (evaluation)
    rx_eval = rx_eval[sync_point::]
    delay = estimate_delay(rx_eval, sps=SPS)
    symbols_est_eval = rx_eval[delay::SPS][0:N_SYMBOLS]
    symbols_est_eval = find_closest_symbol(symbols_est_eval, torch.from_numpy(pam_symbols))

    # Count symbol errors
    symbol_error_rate = torch.sum(torch.logical_not(torch.eq(symbols_est_eval, tx_symbols))) / len(tx_symbols)
    print(f"SER with optimized pulse: {symbol_error_rate}")

    # FIXME: Implement the EsN0 dB theory (same as original code)
    # ... (Rest of the code for EsN0 calculation and plotting remains the same)

    # Plot a subsegment of the transmitted signal and the pulse
    fig, ax = plt.subplots(nrows=2)
    ax[0].plot(t, g)
    ax[0].grid()
    ax[0].set_title('Initial Pulse Shaper')

    time_slice = slice(np.argmax(g)-1, np.argmax(g)-1 + 16 * SPS)
    ax[1].plot(x[time_slice], label='Tx signal (initial)')
    ax[1].plot(y[time_slice], label='Received signal (initial)')
    ax[1].plot(x_eval[time_slice], label='Tx signal (optimized)')
    ax[1].plot(y_eval[time_slice], label='Received signal (optimized)')
    ax[1].grid()
    ax[1].legend()
    ax[1].set_title('Slice of signals')
    plt.tight_layout()

    # Plot distribution of symbols and plot symbol error rate
    fig, ax = plt.subplots()
    ax.hist(rx_eval.detach().cpu().numpy()[::SPS], bins=100)
    ax.set_title('Distribution of signal after matched filter (optimized)')

    plt.show()


if __name__ == "__main__":
    main()


