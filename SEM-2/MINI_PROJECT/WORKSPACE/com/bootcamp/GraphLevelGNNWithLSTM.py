import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GraphLevelGNNWithLSTM(torch.nn.Module):
    def __init__(self, in_channels, h1=71, h2=82, out_channels=1, lstm_hidden_size=32, lstm_layers=1):
        super(GraphLevelGNNWithLSTM, self).__init__()
        self.conv1 = GCNConv(in_channels, h1)
        self.conv2 = GCNConv(h1, h2)
        self.out = GCNConv(h2, out_channels)

        # Define LSTM
        self.lstm = nn.LSTM(input_size=h2, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # GCN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Reshape for LSTM (batch_size, seq_len, feature_dim)
        x_lstm_input = x.unsqueeze(0)  # Assuming a single graph (batch_size=1)

        # LSTM layer
        lstm_out, _ = self.lstm(x_lstm_input)
        lstm_out = lstm_out[:, -1, :]  # Take the last output of the LSTM

        # Fully connected layer for final output
        x_out = self.fc(lstm_out)
        return x_out
