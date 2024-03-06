import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PulseShapeNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PulseShapeNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()  # Use Tanh to ensure the pulse shape can be both positive and negative
        )

    def forward(self, x):
        return self.network(x)