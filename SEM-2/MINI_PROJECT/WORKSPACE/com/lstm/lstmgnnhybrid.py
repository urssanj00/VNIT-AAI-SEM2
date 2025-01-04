import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

# Load the dataset
data_set_path = "C:/Sanjeev/VNIT_CLASSES/mini-proj-dataset/pm2_5/Johannesburg_Westdene_12.csv"
data = pd.read_csv(data_set_path)

# Convert timestamp to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Calculate cyclic time-of-day features before setting timestamp as the index
time_in_seconds = data['timestamp'].dt.hour * 3600 + \
                  data['timestamp'].dt.minute * 60 + \
                  data['timestamp'].dt.second
seconds_in_day = 24 * 3600
timestamps_sin = np.sin(2 * np.pi * time_in_seconds / seconds_in_day)  # Cyclic encoding (sine)
timestamps_cos = np.cos(2 * np.pi * time_in_seconds / seconds_in_day)  # Cyclic encoding (cosine)

# Add cyclic time features to the dataset
data['sin_time'] = timestamps_sin
data['cos_time'] = timestamps_cos

# Set timestamp as the index
data.set_index('timestamp', inplace=True)

# Display the first few rows of the updated dataset
print(data.head())

# Combine cyclic time features into an array
time_features = np.stack([timestamps_sin, timestamps_cos], axis=1)
print(time_features)

# Select features (X) and target (y)
features = ['sensor_id', 'temperature', 'humidity', 'timestamp', 'longitude', 'latitude']
target_column = 'pm2p5'


# Select relevant columns
columns_of_interest = ['sensor_id', 'temperature', 'humidity', 'timestamp', 'longitude', 'latitude']

# Find unique locations
unique_locations = data[columns_of_interest].drop_duplicates(subset=['latitude', 'longitude'])

# Reset index for better readability
unique_locations.reset_index(drop=True, inplace=True)

# Display the unique locations
print(f"unique_locations:{unique_locations}")


# Define edges based on proximity (e.g., k-nearest neighbors)
# For simplicity, we'll connect each node to every other node (fully connected graph)
edge_index = []
num_nodes = len(unique_locations)
for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j:
            edge_index.append([i, j])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # Shape [2, num_edges]

# Create PyTorch Geometric Data object
graph = Data(edge_index=edge_index)



# Extract coordinates
coordinates = np.array([[loc['latitude'], loc['longitude']] for loc in locations])

# Use k-nearest neighbors
k = 2  # Number of neighbors
nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(coordinates)
distances, indices = nbrs.kneighbors(coordinates)

# Build edge index
edge_index = []
for i, neighbors in enumerate(indices):
    for neighbor in neighbors:
        if i != neighbor:
            edge_index.append([i, neighbor])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
graph = Data(edge_index=edge_index)







# Assuming you have `num_time_steps` time steps
num_time_steps = 10
num_nodes = len(locations)
input_features = 2  # [temperature, humidity]

# Create a tensor of shape [num_time_steps, num_nodes, input_features]
# For demonstration, we'll use random data
x = torch.randn(num_time_steps, num_nodes, input_features)


class GNN_LSTM_Hybrid(nn.Module):
    def __init__(self, gnn_in_features, gnn_out_features, lstm_hidden_size, lstm_num_layers, num_nodes):
        super(GNN_LSTM_Hybrid, self).__init__()

        self.num_nodes = num_nodes

        # Define GNN layers
        self.gnn = pyg_nn.GCNConv(gnn_in_features, gnn_out_features)
        self.activation = nn.ReLU()

        # Define LSTM layers
        self.lstm = nn.LSTM(input_size=gnn_out_features,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            batch_first=True)

        # Output layer to predict pm2.5
        self.output_layer = nn.Linear(lstm_hidden_size, 1)  # Predicting a single value per node per time step

    def forward(self, graph, x):
        """
        Parameters:
        - graph: PyTorch Geometric Data object containing edge_index.
        - x: Tensor of shape [num_time_steps, num_nodes, gnn_in_features].

        Returns:
        - predictions: Tensor of shape [num_time_steps, num_nodes, 1].
        """
        num_time_steps, num_nodes, _ = x.size()

        # Initialize a list to collect GNN outputs for each time step
        gnn_outputs = []

        for t in range(num_time_steps):
            # Extract node features for time step t
            x_t = x[t]  # Shape: [num_nodes, gnn_in_features]

            # Apply GNN
            gnn_out = self.gnn(x_t, graph.edge_index)  # Shape: [num_nodes, gnn_out_features]
            gnn_out = self.activation(gnn_out)
            gnn_outputs.append(gnn_out)

        # Stack GNN outputs to form a sequence
        # Shape: [num_time_steps, num_nodes, gnn_out_features]
        gnn_seq = torch.stack(gnn_outputs, dim=0)

        # Prepare LSTM input
        # Reshape to [batch_size=num_nodes, seq_len=num_time_steps, input_size=gnn_out_features]
        lstm_input = gnn_seq.permute(1, 0, 2)

        # Pass through LSTM
        lstm_out, _ = self.lstm(lstm_input)  # Shape: [num_nodes, num_time_steps, lstm_hidden_size]

        # Reshape back to [num_time_steps, num_nodes, lstm_hidden_size]
        lstm_out = lstm_out.permute(1, 0, 2)

        # Apply output layer
        predictions = self.output_layer(lstm_out)  # Shape: [num_time_steps, num_nodes, 1]

        return predictions


# Example Usage

# Hyperparameters
gnn_in_features = 2  # [temperature, humidity]
gnn_out_features = 16
lstm_hidden_size = 32
lstm_num_layers = 2
num_nodes = len(locations)

# Initialize the model
model = GNN_LSTM_Hybrid(gnn_in_features, gnn_out_features, lstm_hidden_size, lstm_num_layers, num_nodes)

# Dummy Data
num_time_steps = 10
x = torch.randn(num_time_steps, num_nodes, gnn_in_features)  # [10, num_nodes, 2]

# Assuming `graph` is defined as in previous sections
predictions = model(graph, x)  # [10, num_nodes, 1]
print(predictions.shape)  # Should output: torch.Size([10, num_nodes, 1])
