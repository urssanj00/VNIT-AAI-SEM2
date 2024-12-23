
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch

class GraphLevelGNN(torch.nn.Module):
    def __init__(self, in_channels, h1=71, h2=82, out_channels=1):
        print(f'{in_channels}/{h1}/{h2}/{out_channels}')
        super(GraphLevelGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, h1)
        self.conv2 = GCNConv(h1, h2)
        self.out = GCNConv(h2, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_conv1 = self.conv1(x, edge_index)
        x_forward = F.relu(x_conv1)
       # x_forward = F.relu(self.conv2(x_forward, edge_index))
        x_forward =  self.conv2(x_forward, edge_index)

        #x = self.conv2(x, edge_index)
        #x = F.relu(x)
        x_forward = self.out(x_forward, edge_index)
        return x_forward
