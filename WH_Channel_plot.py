import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.signal import freqz

# Define the FIR filters
isi_filter1 = np.array([1.0, 0.3, 0.1])
isi_filter2 = np.array([1.0, -0.2, 0.02])

# Frequency response of the first FIR filter
w1, h1 = freqz(isi_filter1, worN=8000)

# Frequency response of the second FIR filter
w2, h2 = freqz(isi_filter2, worN=8000)

# Define the non-linearity
def non_linear_function(x):
    return 1.0 * x + 0.2 * x**2 + (-0.1) * x**3

# Generate a range of input values for the non-linear function
x = np.linspace(-2, 2, 400)
y = non_linear_function(x)

# Plot the frequency responses of the linear filters and the non-linearity

fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plot the first linear block
axs[0].plot(w1 / np.pi, 20 * np.log10(abs(h1)), 'b')
axs[0].set_title('Frequency Response of First Linear Block (FIR filter1)')
axs[0].set_xlabel('Normalized Frequency (×π rad/sample)')
axs[0].set_ylabel('Magnitude (dB)')
axs[0].grid()

# Plot the second linear block
axs[1].plot(w2 / np.pi, 20 * np.log10(abs(h2)), 'r')
axs[1].set_title('Frequency Response of Second Linear Block (FIR filter2)')
axs[1].set_xlabel('Normalized Frequency (×π rad/sample)')
axs[1].set_ylabel('Magnitude (dB)')
axs[1].grid()

# Plot the non-linear function
axs[2].plot(x, y, 'g')
axs[2].set_title('Non-Linearity (3rd Order Polynomial)')
axs[2].set_xlabel('Input')
axs[2].set_ylabel('Output')
axs[2].grid()

plt.tight_layout()
plt.show()