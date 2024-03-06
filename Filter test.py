import torch
import torch.nn as nn
import torch.optim as optim

class TrainableFilter(nn.Module):
    def __init__(self, num_taps=31):
        super().__init__()
        self.coeffs = nn.Parameter(torch.rand(num_taps, requires_grad=True))
    
    def forward(self, x):
        return torch.nn.functional.conv1d(x.unsqueeze(1), self.coeffs.unsqueeze(0)).squeeze(1)

def train_filter(filter, signal_tensor, desired_signal_tensor, num_epochs=1000, learning_rate=0.001):
    optimizer = torch.optim.Adam(filter.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Filter the signal
        filtered_signal = filter(signal_tensor)
        
        # Compute loss and update filter coefficients
        loss = loss_function(filtered_signal, desired_signal_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

# Example usage
trainable_filter = TrainableFilter()
train_filter(trainable_filter, input_signal, target_output_signal)