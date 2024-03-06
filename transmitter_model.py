import torch
import torch.nn as nn
import torch.optim as optim

class TransmitterNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TransmitterNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)  # Hidden dimension is hardcoded for simplicity
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x 

def optimize_transmitter(tx_symbols, N_SYMBOLS, pam_symbols):
    # Create a mapping from PAM symbols to non-negative integers
    symbol_to_index = {symbol: index for index, symbol in enumerate(sorted(pam_symbols))}
    index_to_symbol = {index: symbol for symbol, index in symbol_to_index.items()}

    # Map tx_symbols to non-negative integer indices
    tx_symbol_indices = torch.tensor([symbol_to_index[symbol] for symbol in tx_symbols], dtype=torch.long)

    # One-hot encode the mapped indices
    symbols_one_hot = torch.nn.functional.one_hot(tx_symbol_indices, num_classes=len(pam_symbols)).float()

    # Initialize your model here (ensure it's defined correctly in your environment)
    model = TransmitterNet(input_dim=len(pam_symbols), output_dim=len(pam_symbols))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert symbols to expected format for training
    for epoch in range(100):  # Number of epochs for training
        optimizer.zero_grad()
        outputs = model(symbols_one_hot)
        loss = criterion(outputs, symbols_one_hot)  # Adjust as needed
        loss.backward()
        optimizer.step()

    # Use the trained model to optimize the symbols
    optimized_indices = torch.argmax(model(symbols_one_hot), dim=1)
    optimized_tx_symbols = torch.tensor([index_to_symbol[index.item()] for index in optimized_indices], dtype=torch.float)

    return optimized_tx_symbols

