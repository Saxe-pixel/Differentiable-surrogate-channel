"""
    Module containing the channel models
"""

import numpy as np
import torch
from torch import types as torch_types
from torchaudio.functional import convolve


class CommunicationChannel(object):
    """
        Parent class
    """
    def __init__(self) -> None:
        pass

    def forward(self, x: torch_types._TensorOrTensors):
        raise NotImplementedError


class AWGNChannel(CommunicationChannel):
    """
        Simplest communication channel that only adds white Gaussian noise
    """
    def __init__(self, snr_db, pulse_energy) -> None:
        super().__init__()
        # Calculate noise std based on specified SNR and the pulse energy
        self.noise_std = np.sqrt(pulse_energy / (2 * 10 ** (snr_db/ 10)))

    def forward(self, x: torch_types._TensorOrTensors):
        return x + self.noise_std * torch.randn_like(x)


class AWGNChannelWithLinearISI(CommunicationChannel):
    """
        Communication channel with intersymbol-interference modeled by an FIR filter.
        Afterwards white Gaussian noise is added.
        ISI FIR is taken from

        A. Caciularu and D. Burshtein, “Unsupervised Linear and Nonlinear Channel Equalization and Decoding Using Variational Autoencoders,”
          IEEE Transactions on Cognitive Communications and Networking, vol. 6, no. 3, pp. 1003–1018, Sep. 2020, doi: 10.1109/TCCN.2020.2990773.


    """
    def __init__(self, snr_db, pulse_energy, samples_pr_symbol, dtype=torch.float64) -> None:
        super().__init__()
        h_isi = [0.2, 0.9, 0.3]  # Simple linear transfer function creating ISI
        h_isi_zeropadded = np.zeros(samples_pr_symbol * (len(h_isi) - 1) + 1)
        h_isi_zeropadded[::samples_pr_symbol] = h_isi
        h_isi_zeropadded = h_isi_zeropadded / np.linalg.norm(h_isi_zeropadded)
        self.isi_filter = torch.Tensor(h_isi_zeropadded).type(dtype)

        # Calculate noise std based on specified SNR and the pulse energy
        self.noise_std = np.sqrt(pulse_energy / (2 * 10 ** (snr_db/ 10)))

    def forward(self, x: torch_types._TensorOrTensors):
        xisi = convolve(x, self.isi_filter, mode='same')
        return xisi + self.noise_std * torch.randn_like(xisi)
