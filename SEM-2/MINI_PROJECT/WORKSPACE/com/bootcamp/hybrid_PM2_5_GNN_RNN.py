import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch_geometric.nn import GCNConv

# Step 1: Load the data
X = np.load('X_data.npy')
y = np.load('y_data.npy')

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Custom dataset class
class PM2_5Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Data split and DataLoader creation
train_size = int(0.8 * len(X_tensor))
test_size = len(X_tensor) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    PM2_5Dataset(X_tensor, y_tensor), [train_size, test_size]
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Step 2: Define the hybrid GNN-RNN model
class HybridGNNRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, gnn_hidden_dim, rnn_hidden_dim, output_dim, num_layers=2):
        super(HybridGNNRNN, self).__init__()
        # GNN layers
        self.gnn1 = GCNConv(input_dim, gnn_hidden_dim)
        self.gnn2 = GCNConv(gnn_hidden_dim, gnn_hidden_dim)

        # RNN layers
        self.rnn = nn.LSTM(gnn_hidden_dim, rnn_hidden_dim, num_layers, batch_first=True)

        # Output layer
        self.fc = nn.Linear(rnn_hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # Apply GNN layers
        x = torch.relu(self.gnn1(x, edge_index))
        x = torch.relu(self.gnn2(x, edge_index))  # x.shape: (num_nodes, gnn_hidden_dim)

        # Reshape for RNN input
        batch_size = x.size(0)
        seq_length = X.shape[1]  # Retrieve the sequence length from data
        x = x.view(batch_size, seq_length, -1)  # (batch, seq, gnn_hidden_dim)

        # Apply RNN
        rnn_out, _ = self.rnn(x)

        # Predict next 7 days
        out = self.fc(rnn_out[:, -1, :])  # Use the last RNN output
        return out

# Step 3: Training and Evaluation
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            # Simulate edge_index for GNN (example, adjust as needed)
            edge_index = torch.randint(0, X_batch.shape[1], (2, X_batch.shape[1] * 2))
            outputs = model(X_batch, edge_index)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Instantiate and train the model
input_dim = X.shape[2]  # Features per timestep
hidden_dim = 128
gnn_hidden_dim = 64
rnn_hidden_dim = 128
output_dim = 7

model = HybridGNNRNN(input_dim, hidden_dim, gnn_hidden_dim, rnn_hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, criterion, optimizer)
