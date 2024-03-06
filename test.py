

# Parameterization of the Pulse Shape
class PulseShape(nn.Module):
    def __init__(self, filter_length, sps):
        super(PulseShape, self).__init__()
        self.coefficients = nn.Parameter(torch.rand(filter_length) * 0.1)
        self.sps = sps

    def forward(self, input_signal):
        # Ensure input_signal is in the correct shape for convolution
        input_signal = input_signal.view(1, 1, -1)
        coefficients = self.coefficients.repeat(self.sps, 1).flatten()
        coefficients = coefficients / torch.norm(coefficients)  # Normalize pulse energy
        return torch.nn.functional.conv1d(input_signal, coefficients.view(1, 1, -1), padding='same').squeeze()

if __name__ == "__main__":
    # Simulation parameters
    SEED = 12345
    N_SYMBOLS = int(1e3)  # Reduced number of symbols for faster optimization
    SPS = 8  # samples-per-symbol (oversampling rate)
    SNR_DB = 4.0  # signal-to-noise ratio in dB
    BAUD_RATE = 10e6  # symbols per second
    FILTER_LENGTH = 33  # Reduced filter length for simplicity

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Generate data
    pam_symbols = torch.tensor([-3.0, -1.0, 1.0, 3.0])
    tx_symbols = torch.tensor(np.random.choice(pam_symbols.numpy(), size=(N_SYMBOLS,), replace=True))

    # Initialize pulse shape
    pulse_shape = PulseShape(FILTER_LENGTH, SPS)
    optimizer = optim.Adam(pulse_shape.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(1000):  # Limited number of epochs for demonstration
        optimizer.zero_grad()
        
        # Generate transmitted signal with current pulse shape
        tx_symbols_up = torch.zeros(N_SYMBOLS * SPS)
        tx_symbols_up[::SPS] = tx_symbols
        x = pulse_shape(tx_symbols_up.float())
        
        # Simulate transmission over the channel
        pulse_energy = torch.norm(pulse_shape.coefficients) ** 2
        channel = AWGNChannel(snr_db=SNR_DB, pulse_energy=pulse_energy.item())
        y = channel.forward(x)
        
        # Apply matched filtering (convolution with time-reversed pulse)
        rx = pulse_shape(torch.flip(y, dims=[0])) / pulse_energy
        
        # Estimate delay and adjust received symbols accordingly
        delay = estimate_delay(rx, sps=SPS)
        symbols_est = rx[delay::SPS][:N_SYMBOLS]
        
        # Calculate loss and backpropagate
        loss = loss_fn(symbols_est, tx_symbols.float())
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    
    # After optimization, you can further process `symbols_est` for symbol decision making
    # and calculate the symbol error rate (SER), as done in your original code.
    
    # Plot the optimized pulse shape
    plt.plot(pulse_shape.coefficients.detach().numpy(), label='Optimized Pulse Shape')
    plt.legend()
    plt.show()
