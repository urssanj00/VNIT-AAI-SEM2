import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

class AdvancedTemporalGraphNetwork(nn.Module):
    def __init__(self, num_features, hidden_channels, num_nodes, dropout=0):
        super(AdvancedTemporalGraphNetwork, self).__init__()
        self.init_params = {
            'num_features': num_features,
            'hidden_channels': hidden_channels,
            'num_nodes': num_nodes,
            'dropout': dropout
        }
        self.feature_reducer = nn.Linear(num_features, hidden_channels)
        self.graph_conv1 = pyg_nn.SAGEConv(hidden_channels, hidden_channels)
        self.graph_conv2 = pyg_nn.SAGEConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.temporal_lstm = nn.LSTM(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=2,
            batch_first=True,
            dropout=dropout
        )

        # Prediction head now outputs only 1 value for pm2p5
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels // 2, 1)  # Output size set to 1
        )

    def forward(self, x, edge_index, time_sequence):
        num_features = self.init_params['num_features']
        num_nodes = self.init_params['num_nodes']
        if x is None:
            x = torch.randn(num_nodes, num_features, device=edge_index.device)
        x = x.float()
        time_sequence = time_sequence.float()
        reduced_features = self.feature_reducer(x)
        spatial_features = F.relu(self.graph_conv1(reduced_features, edge_index))
        spatial_features = F.relu(self.graph_conv2(spatial_features, edge_index))
        time_sequence_reduced = self.feature_reducer(time_sequence)
        lstm_out, _ = self.temporal_lstm(time_sequence_reduced)
        attn_out, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)
        predictions = self.prediction_head(attn_out[:, -1, :])  # Predict only pm2p5
        return predictions

    def get_init_params(self):
        return self.init_params


'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

class AdvancedTemporalGraphNetwork(nn.Module):
    def __init__(self, num_features, hidden_channels, num_nodes, dropout=0):
        super(AdvancedTemporalGraphNetwork, self).__init__()
        self.init_params = {
            'num_features': num_features,
            'hidden_channels': hidden_channels,
            'num_nodes': num_nodes,
            'dropout': dropout
        }
        self.feature_reducer = nn.Linear(num_features, hidden_channels)
        self.graph_conv1 = pyg_nn.SAGEConv(hidden_channels, hidden_channels)
        self.graph_conv2 = pyg_nn.SAGEConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.temporal_lstm = nn.LSTM(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=2,
            batch_first=True,
            dropout=dropout
        )

        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels//2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels//2, num_features)
        )
    def forward(self, x, edge_index, time_sequence):
        num_features = self.init_params['num_features']
        num_nodes = self.init_params['num_nodes']
        if x is None:
          x = torch.randn(num_nodes,num_features, device = edge_index.device)
        x = x.float()
        time_sequence = time_sequence.float()
        reduced_features = self.feature_reducer(x)
        spatial_features = F.relu(self.graph_conv1(reduced_features, edge_index))
        spatial_features = F.relu(self.graph_conv2(spatial_features, edge_index))
        time_sequence_reduced = self.feature_reducer(time_sequence)
        lstm_out, _ = self.temporal_lstm(time_sequence_reduced)
        attn_out, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)
        predictions = self.prediction_head(attn_out[:, -1, :])
        return predictions

    def get_init_params(self):
        return self.init_params
'''