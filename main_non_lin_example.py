"""
    Example of the additive white Gaussian noise (AWGN) channel 
    with/without inter-symbol interference (ISI)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import torchaudio.functional as taf
from commpy.filters import rrcosfilter

from lib.utility import estimate_delay, find_closest_symbol
from lib.channels import WienerHammersteinISIChannel


if __name__ == "__main__":
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

    # Construct pulse shape
    t, g = rrcosfilter(FILTER_LENGTH, 0.5, Ts, fs)
    g /= np.linalg.norm(g)  # ensure that pulse as unit norm
    pulse = torch.from_numpy(g)

    # Calculate some needed statistics on the pulse - used later for synchronization
    gg = np.convolve(g, g[::-1])
    pulse_energy = np.max(gg)
    print(f"Energy of pulse is: {pulse_energy}")
    sync_point = np.argmax(gg)

    # Transmit - up-sample the symbols and apply pulse
    tx_symbols_up = torch.zeros((N_SYMBOLS * SPS, ), dtype=torch.double)
    tx_symbols_up[0::SPS] = tx_symbols
    x = taf.convolve(tx_symbols_up, pulse)

    # Apply the "unknown" channel
    channel = WienerHammersteinISIChannel(snr_db=SNR_DB, pulse_energy=pulse_energy, samples_pr_symbol=SPS,
                                          non_linear_coefficients=(1.0, 0.2, -0.1))
    y = channel.forward(x)

    # Apply receiver
    rx = taf.convolve(y, torch.flip(pulse, (0,))) / pulse_energy

    # Correct for the pulse-convolutions
    rx = rx[sync_point::]

    # Decision-making - if channel has introduced a delay we need to correct for that
    # NB! Very important during training/optimization
    delay = estimate_delay(rx, sps=SPS)
    print(f"Delay was estimated to be: {delay}")
    symbols_est = rx[delay::SPS]  # pick out every SPS samples to get the symbols
    symbols_est = symbols_est[0:N_SYMBOLS]  # truncate the tail
    symbols_est = find_closest_symbol(symbols_est, torch.from_numpy(pam_symbols))

    # Count symbol errors
    symbol_error_rate = torch.sum(torch.logical_not(torch.eq(symbols_est, tx_symbols))) / len(tx_symbols)
    print(f"SER: {symbol_error_rate:.3e}")

    # Plot a subsegment of the transmitted signal and the pulse
    fig, ax = plt.subplots(nrows=2)
    ax[0].plot(t, g)
    ax[0].grid()
    ax[0].set_title('Pulse shaper')

    time_slice = slice(np.argmax(g)-1, np.argmax(g)-1 + 16 * SPS)
    ax[1].plot(x[time_slice], label='Tx signal')
    ax[1].plot(y[time_slice], label='Received signal')
    ax[1].grid()
    ax[1].legend()
    ax[1].set_title('Slice of signals')
    plt.tight_layout()

    # Plot distribution of symbols and plot symbol error rate
    fig, ax = plt.subplots()
    ax.hist(rx.detach().cpu().numpy()[::SPS], bins=100)
    ax.set_title('Distribution of signal after matched filter')

    plt.show()
