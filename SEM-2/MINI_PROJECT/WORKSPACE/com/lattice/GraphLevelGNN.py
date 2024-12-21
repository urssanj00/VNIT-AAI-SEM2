import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch
import torch.nn as nn
import torch.optim as optim

class GraphLevelGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphLevelGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        print(f"x shape: {x.shape}, edge_index: {edge_index.size()}")

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        print()
        return global_mean_pool(x, batch=None)  # Mean pooling to get graph-level embedding
