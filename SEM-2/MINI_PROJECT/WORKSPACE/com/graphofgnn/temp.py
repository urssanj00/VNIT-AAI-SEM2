# Import necessary libraries
import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from com.dataprep.PropertiesConfig import PropertiesConfig as PC
from GNNCustomized import GNNCustomized as GNN

# Create an instance of PropertiesConfig
properties_config = PC("sensor-data.properties")
properties = properties_config.get_properties_config()
data_set_dir = properties['data_set_path']
plot_dir = properties['plot_path']

# Step 0: Load the file list with directory

def load_file_list():
    station_directory = f"{data_set_dir}"
    file_list = []
    try:
        files = os.listdir(station_directory)
        filtered_files = [f for f in files if 'station_list' not in f.lower()]
        for file in filtered_files:
            file_list.append(f'{station_directory}/{file}')
    except FileNotFoundError:
        print("Directory not found:", station_directory)
    print('File list loaded.')
    return file_list

# Step 1: Load and preprocess the data
def load_data(file_list):
    dataframes = []
    for filepath in file_list:
        try:
            with open(filepath, 'r') as file:
                df = pd.read_csv(file)
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    combined_data = pd.concat(dataframes, ignore_index=True)
    print('Data combined.')
    return combined_data

# Add lag features for temporal prediction
def add_lag_features(df, target_column, lag_days=7):
    for lag in range(1, lag_days + 1):
        df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
    df.dropna(inplace=True)
    print('Lag features added.')
    return df

# Step 2: Normalize features
def normalize_features(df, feature_columns):
    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    print('Features normalized.')
    return df, scaler

# Step 3: Create a graph with temporal data
def create_graph_with_temporal_data(df, lag_days, distance_threshold=0.01):
    G = nx.Graph()
    for i, row in df.iterrows():
        features = row[['temperature', 'humidity', 'longitude', 'latitude']].tolist()
        lag_features = row[[f'pm2p5_lag_{lag}' for lag in range(1, lag_days + 1)]].tolist()
        G.add_node(i, features=features + lag_features, target=row['pm2p5'])

    for i, row_i in df.iterrows():
        for j, row_j in df.iterrows():
            if i != j:
                distance = np.sqrt((row_i['longitude'] - row_j['longitude'])**2 + (row_i['latitude'] - row_j['latitude'])**2)
                if distance < distance_threshold:
                    G.add_edge(i, j)

    print(f"Graph created with {len(G.edges)} edges.")
    visualize_graph(G, df)
    return G

def visualize_graph(G, df):
    positions = {i: (row['longitude'], row['latitude']) for i, row in df.iterrows()}
    plt.figure(figsize=(10, 7))
    nx.draw(
        G,
        pos=positions,
        with_labels=True,
        node_color='lightblue',
        edge_color='gray',
        node_size=300,
        font_size=8
    )
    plt.title("Graph Representation with Temporal Data")
    plt.savefig(f'{plot_dir}/graph.png')
    plt.close()

# Step 4: Convert to PyTorch Geometric Data
def graph_to_pyg_data(G):
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    features = graph_to_pyg_data_features(G)
    num_nodes = features.size(0)

    valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
    edge_index = edge_index[:, valid_mask]

    target = torch.tensor([G.nodes[node]['target'] for node in G.nodes], dtype=torch.float)
    data = Data(x=features, edge_index=edge_index, y=target)

    print(f"Converted graph to PyTorch Geometric format. Nodes: {num_nodes}, Edges: {edge_index.size(1)}")
    return data

def graph_to_pyg_data_features(G):
    first_node = list(G.nodes())[0]
    feature_dim = len(G.nodes[first_node]['features'])
    node_features = np.zeros((len(G.nodes), feature_dim))

    for idx, node in enumerate(G.nodes):
        node_features[idx] = G.nodes[node]['features']

    features = torch.tensor(node_features, dtype=torch.float32)
    return features

# Step 6: Train the model
def train_model(data, model, epochs=100, lr=0.01):
    print('Starting model training...')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    model.train()

    epoch_counts, loss_values = [], []
    predictions, actual_values = [], []

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        epoch_counts.append(epoch)
        loss_values.append(loss.item())

        if epoch % 10 == 0:
            predictions.append(out.detach().cpu().numpy())
            actual_values.append(data.y.detach().cpu().numpy())
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    save_plot(epoch_counts, loss_values, "epoch_loss.png", "Epochs", "Loss", "Training Loss")
    save_scatter_plot(actual_values, predictions, "actual_vs_predicted.png")
    return model

def save_plot(x, y, filename, xlabel, ylabel, title):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{plot_dir}/{filename}')
    plt.close()
    print(f"Plot saved: {filename}")

def save_scatter_plot(actual_values, predictions, filename):
    plt.figure(figsize=(10, 6))
    for i in range(len(predictions)):
        plt.scatter(actual_values[i], predictions[i], label=f"Epoch {i * 10}")
    plt.plot(
        [min(actual_values[0]), max(actual_values[0])],
        [min(actual_values[0]), max(actual_values[0])],
        color='red', linestyle='--', label='Perfect Fit'
    )
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values")
    plt.legend()
    plt.grid()
    plt.savefig(f'{plot_dir}/{filename}')
    plt.close()
    print(f"Scatter plot saved: {filename}")

# Step 7: Predict for the next 7 days
def predict_next_7_days(data, model, scaler):
    model.eval()
    predictions = []
    with torch.no_grad():
        input_data = data.x[-1]

        for _ in range(7):
            input_dim = model.conv1.in_channels
            if input_data.shape[0] < input_dim:
                input_data = torch.cat([input_data, torch.zeros(input_dim - input_data.shape[0])])
            elif input_data.shape[0] > input_dim:
                input_data = input_data[:input_dim]

            temp_data = Data(x=input_data.unsqueeze(0), edge_index=torch.zeros((2, 0), dtype=torch.long))
            prediction = model(temp_data).item()
            predictions.append(prediction)
            input_data = torch.cat([input_data[4:], torch.tensor([prediction])])

    target_scaler = MinMaxScaler()
    target_scaler.min_, target_scaler.scale_ = scaler.min_[-1:], scaler.scale_[-1:]
    predictions = target_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    print('Next 7 days predictions:', predictions)
    return predictions

# Main program
if __name__ == "__main__":
    csv_list = load_file_list()
    combined_data = load_data(csv_list)
    feature_columns = ['temperature', 'humidity', 'longitude', 'latitude']
    target_column = 'pm2p5'
    combined_data = add_lag_features(combined_data, target_column)
    combined_data, scaler = normalize_features(combined_data, feature_columns + [f'{target_column}_lag_{i}' for i in range(1, 8)])

    G = create_graph_with_temporal_data(combined_data, lag_days=7, distance_threshold=0.2)
    data = graph_to_pyg_data(G)

    input_dim = len(feature_columns) + 7
    hidden_dim = 16
    output_dim = 1
    model = GNN(input_dim, hidden_dim, output_dim)
    trained_model = train_model(data, model)
    predict_next_7_days(data, trained_model, scaler)
