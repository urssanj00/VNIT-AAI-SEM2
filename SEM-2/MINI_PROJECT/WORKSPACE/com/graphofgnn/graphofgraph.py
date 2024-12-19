# Import necessary libraries
import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from com.dataprep.PropertiesConfig import PropertiesConfig as PC
import os

# Create an instance of PropertiesConfig
properties_config = PC("sensor-data.properties")
properties = properties_config.get_properties_config()
data_set_dir = properties['data_set_path']


# Step 0: Load the file name with DIR in a file_list
def load_file_list():
    station_directory = f"{data_set_dir}"
    # List of files
    file_list = []  # Add all 19 file names here
    try:
        files = os.listdir(station_directory)
        filtered_files = [f for f in files if 'station_list' not in f.lower()]

        print("Files in", station_directory, ":")
        for file in filtered_files:
            file_list.append(f'{station_directory}/{file}')
    except FileNotFoundError:
        print("Directory not found:", station_directory)

    print('0. load_file_list done')
    return file_list

# Step 1: Load and preprocess the data
def load_data(file_list):
    dataframes = []
    print(0)
    for filepath in file_list:
        print(f'1. {filepath}')
        try:
            with open(filepath, 'r') as file:
                df = pd.read_csv(file)
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue
    combined_data = pd.concat(dataframes, ignore_index=True)
    print('1. load_data done')

    return combined_data

# Add lag features for temporal prediction
def add_lag_features(df, target_column, lag_days=7):
    for lag in range(1, lag_days + 1):
        df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
    df.dropna(inplace=True)
    print('Added lag features for temporal data.')
    return df

# Step 2: Normalize features
def normalize_features(df, feature_columns):
    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    print('2. normalize_features done')

    return df, scaler

# Step 3: Create a graph with temporal data
def create_graph_with_temporal_data(df, lag_days):
    G = nx.Graph()
    for i, row in df.iterrows():
        features = row[['temperature', 'humidity', 'longitude', 'latitude']].tolist()
        lag_features = row[[f'pm2p5_lag_{lag}' for lag in range(1, lag_days + 1)]].tolist()
        G.add_node(i, features=features + lag_features, target=row['pm2p5'])

    # Add edges based on geographical proximity
    for i, row_i in df.iterrows():
        for j, row_j in df.iterrows():
            if i != j:
                distance = np.sqrt((row_i['longitude'] - row_j['longitude'])**2 + (row_i['latitude'] - row_j['latitude'])**2)
                if distance < 0.01:  # Example threshold
                    G.add_edge(i, j)

    print('3. create_graph_with_temporal_data done')
    return G


# Step 4: Convert to PyTorch Geometric Data
def graph_to_pyg_data(G):
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    features = graph_to_pyg_data_features(G)
    num_nodes = features.size(0)

    # Ensure edge_index only contains valid indices
    valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
    edge_index = edge_index[:, valid_mask]

    target = torch.tensor([G.nodes[node]['target'] for node in G.nodes], dtype=torch.float)
    data = Data(x=features, edge_index=edge_index, y=target)

    # Debugging info
    print('4. graph_to_pyg_data done')
    print("Edge Index Shape:", edge_index.shape)
    print("Number of Nodes:", num_nodes)

    return data



def graph_to_pyg_data_features(G):
    # Get the feature dimension from the first node
    first_node = list(G.nodes())[0]
    feature_dim = len(G.nodes[first_node]['features'])

    # Pre-allocate a numpy array with the correct shape
    num_nodes = len(G.nodes)
    node_features = np.zeros((num_nodes, feature_dim))

    # Fill the array with features
    for idx, node in enumerate(G.nodes):
        node_features[idx] = G.nodes[node]['features']

    # Convert to tensor
    features = torch.tensor(node_features, dtype=torch.float32)
    print("Feature tensor shape:", features.shape)
    return features


# Step 5: Define the GNN model
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        print('5.0 GNN.__init__ done')

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = x.squeeze(-1)  # Ensure output is 1D
        print('5.1 GNN.forward done')
        return x

# Step 6: Train the model
def train_model(data, model, epochs=200, lr=0.01):
    print('6.0 train_model start')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'6.0 train_model Epoch:{epoch} Loss:{loss.item()}')
    print('6. train_model done')
    return model

# Step 7: Predict for the next 7 days

def predict_next_7_days(data, model, scaler):
    model.eval()
    predictions = []
    with torch.no_grad():
        # Start with the most recent node's features
        input_data = data.x[-1]  # Shape: [input_dim]

        for _ in range(7):
            # Ensure input_data matches input_dim of the model
            input_dim = model.conv1.in_channels
            if input_data.shape[0] < input_dim:
                # Pad with zeros if input_data has fewer features
                input_data = torch.cat([input_data, torch.zeros(input_dim - input_data.shape[0])])
            elif input_data.shape[0] > input_dim:
                # Truncate if input_data has more features
                input_data = input_data[:input_dim]

            # Create a new Data object for the current step
            temp_data = Data(x=input_data.unsqueeze(0), edge_index=torch.zeros((2, 0), dtype=torch.long))

            # Predict the next value
            prediction = model(temp_data).item()
            predictions.append(prediction)

            # Update input_data with the new prediction
            input_data = torch.cat([input_data[4:], torch.tensor([prediction])])

    # Rescale predictions back to the original range using only the scaler for the target
    target_scaler = MinMaxScaler()
    target_scaler.min_, target_scaler.scale_ = scaler.min_[-1:], scaler.scale_[-1:]  # Use the last feature (pm2p5)
    predictions = target_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    print('Predicted values for the next 7 days:', predictions)
    return predictions



# Main program
if __name__ == "__main__":
    csv_list = load_file_list()

    # Load and preprocess data
    combined_data = load_data(csv_list)
    feature_columns = ['temperature', 'humidity', 'longitude', 'latitude']
    target_column = 'pm2p5'
    combined_data = add_lag_features(combined_data, target_column)
    combined_data, scaler = normalize_features(combined_data, feature_columns + [f'{target_column}_lag_{i}' for i in range(1, 8)])

    # Create graph with temporal data
    G = create_graph_with_temporal_data(combined_data, lag_days=7)

    # Convert to PyTorch Geometric Data
    data = graph_to_pyg_data(G)

    # Define and train the GNN model
    input_dim = len(feature_columns) + 7  # Include lag features
    hidden_dim = 16
    output_dim = 1
    model = GNN(input_dim, hidden_dim, output_dim)
    trained_model = train_model(data, model)

    # Predict for the next 7 days
    predict_next_7_days(data, trained_model, scaler)
