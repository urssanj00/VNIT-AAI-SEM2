import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data_set_path = "C:/Sanjeev/VNIT_CLASSES/mini-proj-dataset/pm2_5/Johannesburg_Westdene_12.csv"
data = pd.read_csv(data_set_path)

# Preprocess the dataset
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)
features = ['sensor_id', 'temperature', 'humidity', 'longitude', 'latitude']
target_column = 'pm2p5'
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[features + [target_column]])



# Define GNN-LSTM Hybrid Model
class GNNLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, gnn_hidden_dim):
        super(GNNLSTMModel, self).__init__()
        self.gnn = GCNConv(input_dim, gnn_hidden_dim)
        self.lstm = nn.LSTM(gnn_hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, seq_length):
        print(f"1. {x.size()}")
       # for i in range(len(x)):
       #     print(f"{i}:{x[i]}")
        gnn_out = self.gnn(x, edge_index)
        print(2)
        lstm_in = gnn_out.unsqueeze(1).repeat(1, seq_length, 1)
        print(3)
        _, (hn, _) = self.lstm(lstm_in)
        print(4)
        out = self.fc(hn[-1])
        print(5)
        return out


# Prepare training and testing datasets
sequence_length = 10
X, y = [], []
for i in range(len(data_scaled) - sequence_length):
    X.append(data_scaled[i:i + sequence_length, :-1])
    y.append(data_scaled[i + sequence_length, -1])
X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Function to construct graph using temperature and humidity
def construct_graph(data):
    G = nx.Graph()
    data_sorted = data.sort_values(by=['temperature', 'humidity'])
    for i, row1 in data_sorted.iterrows():
        for j, row2 in data_sorted.iterrows():
            if i != j:
                G.add_edge(i, j)  # Connect all sorted nodes in sequence
    return from_networkx(G)


# Construct the graph using only the training data subset
train_data = data.iloc[:len(X_train)]  # Use only the training subset
graph_data = construct_graph(train_data)
# Now ensure the edge_index corresponds to the correct subset of nodes (same number as X_train)
edge_index = graph_data.edge_index

# Check if the edge_index contains valid indices (within bounds)
num_nodes = X_train.shape[0]  # Number of training samples (nodes in the graph)

# Ensure that the node indices in edge_index are within the range [0, num_nodes-1]
assert edge_index.max() < num_nodes, f"Edge index out of bounds: {edge_index.max()} >= {num_nodes}"


# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Now you can safely pass the data and graph into the model
X_train_graph = torch.tensor(X_train_tensor, dtype=torch.float32)  # Ensure using the training data



# Initialize and train the model
input_dim = len(features)
hidden_dim = 50
gnn_hidden_dim = 32
output_dim = 1
num_layers = 2
model = GNNLSTMModel(input_dim, hidden_dim, output_dim, num_layers, gnn_hidden_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
# Training Loop
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Use only the training subset of the data
    edge_index = graph_data.edge_index
    #X_train_graph = torch.tensor(X_train_tensor, dtype=torch.float32)  # Ensure it's the training data
    X_train_graph = X_train_tensor.clone().detach()

    print(f"{X_train_graph.size()}:{edge_index.size()}")
    # Forward pass
    y_pred = model(X_train_graph, edge_index, sequence_length)

    # Ensure y_train_tensor is the correct shape
    y_train_tensor = y_train_tensor.view(-1)  # Flatten to match the output shape
    loss = criterion(y_pred.squeeze(), y_train_tensor)  # Ensure no shape mismatch

    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(torch.tensor(data_scaled[-len(y_test):, :-1], dtype=torch.float32), graph_data.edge_index, sequence_length)
y_pred_rescaled = scaler.inverse_transform(np.concatenate((np.zeros((len(y_pred), len(features))), y_pred.numpy()), axis=1))[:, -1]
y_test_rescaled = scaler.inverse_transform(np.concatenate((np.zeros((len(y_test), len(features))), y_test.reshape(-1, 1)), axis=1))[:, -1]

# Save predictions
data['Predicted_PM2.5'] = np.nan
test_indices = data.index[-len(y_test_rescaled):]
data.loc[test_indices, 'Predicted_PM2.5'] = y_pred_rescaled
output_path = "C:/Sanjeev/VNIT_CLASSES/mini-proj-dataset/pm2_5/Johannesburg_with_predictions.csv"
data.to_csv(output_path)
print(f"Predictions saved to: {output_path}")

# Confusion matrix
threshold = 0.5
y_pred_binary = (y_pred_rescaled > threshold).astype(int)
y_test_binary = (y_test_rescaled > threshold).astype(int)
cm = confusion_matrix(y_test_binary, y_pred_binary)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
