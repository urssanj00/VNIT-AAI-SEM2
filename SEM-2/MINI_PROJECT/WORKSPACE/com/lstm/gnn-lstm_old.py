import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx
from lstm_pytorch import *

# Define the GNN-LSTM Hybrid Model
class GNNLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, gnn_hidden_dim):
        super(GNNLSTMModel, self).__init__()
        # GNN Layer
        self.gnn = GCNConv(input_dim, gnn_hidden_dim)

        # LSTM Layer
        self.lstm = nn.LSTM(gnn_hidden_dim, hidden_dim, num_layers, batch_first=True)

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # Pass through GNN
        gnn_out = self.gnn(x, edge_index)

        # Reshape for LSTM (Batch, Sequence, Features)
        lstm_in = gnn_out.unsqueeze(1).repeat(1, sequence_length, 1)

        # Pass through LSTM
        _, (hn, _) = self.lstm(lstm_in)

        # Fully Connected Layer
        out = self.fc(hn[-1])
        return out


# Construct a Graph
def construct_graph(data):
    G = nx.Graph()
    for i, row in data.iterrows():
        G.add_node(i, features=row[features].values)
    # Add edges based on spatial proximity (example: distance threshold)
    for i, node1 in enumerate(data.index):
        for j, node2 in enumerate(data.index):
            if i != j and haversine_distance(data.loc[node1], data.loc[node2]) < 1.0:  # Example threshold
                G.add_edge(node1, node2)
    return from_networkx(G)


# Example Distance Function
def haversine_distance(row1, row2):
    # Use longitude and latitude to calculate distance
    lon1, lat1, lon2, lat2 = map(np.radians, [row1['longitude'], row1['latitude'], row2['longitude'], row2['latitude']])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))  # Earth radius in km


# Prepare Graph Data
graph_data = construct_graph(data)

# Prepare Model and Data
input_dim = len(features)
hidden_dim = 50
gnn_hidden_dim = 32
output_dim = 1
num_layers = 2

model = GNNLSTMModel(input_dim, hidden_dim, output_dim, num_layers, gnn_hidden_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 50
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        edge_index = graph_data.edge_index
        X_graph = graph_data.x.float()

        y_pred = model(X_graph, edge_index)
        loss = criterion(y_pred.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}")

# Evaluation follows a similar pattern as the LSTM.
