import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch
import torch.nn as nn
import torch.optim as optim

class GraphLevelGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        print(f'{in_channels}/{hidden_channels}/{out_channels}')
        super(GraphLevelGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
       # print(f"x shape: {x.shape}, edge_index: {edge_index.size()}")
       # print(f'0. original x:{x}')

        x = self.conv1(x, edge_index)
       # print(f'1. conv1 x:{x}')

        x = F.relu(x)
       # print(f'2. relu x:{x}')

        x = self.conv2(x, edge_index)
       # print(f'2. conv2 x:{x}')

        return x
