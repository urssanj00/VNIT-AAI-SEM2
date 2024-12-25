import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import torch

class GraphLevelGNN(torch.nn.Module):
    def __init__(self, in_channels, h1=71, h2=82, out_channels=1):
        super(GraphLevelGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, h1)
        self.conv2 = GCNConv(h1, h2)
        self.fc = GCNConv(h2, out_channels)  # Linear layer for graph-level prediction

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        print(f"Input shape: {x.shape}, Edge index shape: {edge_index.shape}") #, Batch shape: {batch.shape}")
        print(f"0. x shape: {x.shape}")

        x = F.relu(self.conv1(x, edge_index))
        print(f"1. x shape: {x.shape}")

        x = F.relu(self.conv2(x, edge_index))
        print(f"2. x shape: {x.shape}")

        # Pooling operation to obtain a single graph representation
        '''
        x = global_mean_pool(x, batch)  # Graph-level representation
        print(f"3. x shape: {x.shape}")

        print(f"Pooled output shape (graph-level): {x.shape}")
        '''
        # Final graph-level prediction
        x = self.fc(x, edge_index)
        print(f"4. x shape: {x.shape}")

        return x
