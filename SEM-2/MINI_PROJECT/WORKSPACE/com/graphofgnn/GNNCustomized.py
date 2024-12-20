import torch
from torch_geometric.nn import GCNConv

# Step 5: Define the GNN model
class GNNCustomized(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNCustomized, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        print('5.0 GNN.__init__ done')

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = x.squeeze(-1)  # Ensure output is 1D for compatibility
        print(f'Model forward pass completed. Output shape: {x.shape}')
        return x
