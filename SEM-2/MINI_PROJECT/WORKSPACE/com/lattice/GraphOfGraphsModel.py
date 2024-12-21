import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GraphOfGraphsModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphOfGraphsModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, graph_data, meta_graph_data):
        # Process the individual graph data (graph-level embedding)
        print(f'graph_data: {graph_data}')
        graph_embedding = self.conv1(graph_data.x, graph_data.edge_index)
        graph_embedding = F.relu(graph_embedding)
        graph_embedding = self.conv2(graph_embedding, graph_data.edge_index)

        # Now, use the meta-graph to aggregate the embeddings of the individual graphs
        meta_embedding = self.conv1(graph_embedding, meta_graph_data.edge_index)
        meta_embedding = F.relu(meta_embedding)
        meta_embedding = self.conv2(meta_embedding, meta_graph_data.edge_index)

        return F.log_softmax(meta_embedding, dim=1)