"""
    Various utility functions
"""

import torch


def estimate_delay(y, sps):
    """
        Estimate delay by the maximum-variance sample method
    """
    nsyms = len(y) // sps  # truncate to an integer num symbols
    yr = torch.reshape(y[0:nsyms*sps], (-1, sps))
    var = torch.var(yr, axis=0)
    max_variance_sample = torch.argmax(var)
    return max_variance_sample


def find_closest_symbol(xhat, constellation):
    """
        For element in vector xhat find the closest element in constellation
        'Closest' is defined in the L2 sense.
    """
    sqdiff = torch.square(torch.absolute(xhat[:, None] - constellation[None, :]))
    min_indices = torch.argmin(sqdiff, axis=1)
    assert (len(min_indices) == len(xhat))
    return constellation[min_indices]
