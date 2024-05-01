import numpy as np
import torch
import matplotlib.pyplot as plt
import torchaudio.functional as taf
from commpy.filters import rrcosfilter
from scipy.special import erfc

from lib.utility import estimate_delay, find_closest_symbol
from lib.channels import AWGNChannel, AWGNChannelWithLinearISI

if __name__ == "__main__":
 # Simulation parameters
    SEED = 12345
    N_SYMBOLS = int(1e5)
    SPS = 8  # samples-per-symbol (oversampling rate)
    SNR_DB_RANGE = np.linspace(0, 9, 10)  # signal-to-noise ratio in dB
    BAUD_RATE = 10e6  # number of symbols transmitted pr. second
    FILTER_LENGTH = 256
    Ts = 1 / (BAUD_RATE)  # symbol length
    fs = BAUD_RATE * SPS
    M = 4  # Modulation order for 4-PAM
    log2M = np.log2(M)

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

    # SER data storage
    experimental_SER_AWGN = []
    experimental_SER_ISI = []
    theoretical_SER = []

    for SNR_DB in SNR_DB_RANGE:
        tx_symbols_up = torch.zeros((N_SYMBOLS * SPS,), dtype=torch.double)
        tx_symbols_up[0::SPS] = tx_symbols
        x = taf.convolve(tx_symbols_up, pulse)

        # AWGN channel simulation
        channel_awgn = AWGNChannel(snr_db=SNR_DB, pulse_energy=1)
        y_awgn = channel_awgn.forward(x)
        rx_awgn = taf.convolve(y_awgn, torch.flip(pulse, (0,))) / 1
        rx_awgn = rx_awgn[np.argmax(np.convolve(g, g[::-1]))::SPS]

        # AWGN channel with ISI simulation
        channel_isi = AWGNChannelWithLinearISI(snr_db=SNR_DB, pulse_energy=1, samples_pr_symbol=SPS, dtype=torch.float64)
        y_isi = channel_isi.forward(x)
        rx_isi = taf.convolve(y_isi, torch.flip(pulse, (0,))) / 1
        rx_isi = rx_isi[np.argmax(np.convolve(g, g[::-1]))::SPS]

        # Symbol estimation and SER calculation
        symbols_est_awgn = find_closest_symbol(rx_awgn[:N_SYMBOLS], torch.from_numpy(pam_symbols))
        symbols_est_isi = find_closest_symbol(rx_isi[:N_SYMBOLS], torch.from_numpy(pam_symbols))

        ser_awgn = torch.mean((symbols_est_awgn != tx_symbols).float()).item()
        ser_isi = torch.mean((symbols_est_isi != tx_symbols).float()).item()

        experimental_SER_AWGN.append(ser_awgn)
        experimental_SER_ISI.append(ser_isi)

        # Theoretical SER for AWGN channel
        SNR_linear = 10 ** (SNR_DB / 10)
        Eb_N0_linear = SNR_linear / np.log2(M)
        Q = lambda x: 0.5 * erfc(x / np.sqrt(2))
        ser_theoretical = 2 * (1 - 1/M) * Q(np.sqrt(2 * np.log2(M) * Eb_N0_linear))
        theoretical_SER.append(ser_theoretical)

    # Plotting results
    plt.figure(figsize=(10, 5))
    plt.semilogy(SNR_DB_RANGE, experimental_SER_AWGN, 'o-', label='AWGN Channel SER')
    plt.semilogy(SNR_DB_RANGE, experimental_SER_ISI, 'x-', label='AWGN with ISI SER')
    plt.semilogy(SNR_DB_RANGE, theoretical_SER, 's-', label='Theoretical SER')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Symbol Error Rate (SER)')
    plt.title('SNR vs. SER Comparison')
    plt.ylim(1e-4, 1e0)  # Setting y-axis limits
    plt.grid(True)
    plt.legend()
    plt.show()
