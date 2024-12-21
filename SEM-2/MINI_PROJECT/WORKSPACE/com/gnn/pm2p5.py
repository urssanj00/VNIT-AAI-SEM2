import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from com.dataprep.PropertiesConfig import PropertiesConfig as PC


properties_config = PC()
# Get properties as a dictionary
properties = properties_config.get_properties_config()
csv_path = properties['data_set_path']
filename = f"{csv_path}/{properties['csv_name']}"
print(f'filename {filename}')
# Step 1: Load Data  j.b_marks_secondary_5.csv
data = pd.read_csv(filename)  # Replace with your dataset path
data['timestamp'] = pd.to_datetime(data['timestamp'])
data = data.sort_values(by=['sensor_id', 'timestamp']).reset_index(drop=True)

# Step 2: Feature Scaling
scaler = MinMaxScaler()
data[['temperature', 'humidity', 'pm2p5']] = scaler.fit_transform(data[['temperature', 'humidity', 'pm2p5']])

# Step 3: Prepare Node Features and Targets
node_features = []
node_targets = []
edge_index = []

# Simulate edges (fully connected graph for all sensors)
unique_sensors = data['sensor_id'].unique()
sensor_to_index = {sensor: idx for idx, sensor in enumerate(unique_sensors)}
print(f'sensor_to_index : {sensor_to_index}')
# Group by sensor_id for node features
for sensor_id, group in data.groupby('sensor_id'):
    group = group.sort_values('timestamp')
    features = group[['temperature', 'humidity', 'pm2p5']].shift(1).dropna()
    targets = group['pm2p5'][1:].values  # Next day pm2p5
    node_features.append(features.values)
    node_targets.append(targets)

# Convert features and targets to tensors
x = torch.tensor(node_features[0], dtype=torch.float)  # First sensor for simplicity
y = torch.tensor(node_targets[0], dtype=torch.float)

# Create edges (fully connected graph)
for i in range(len(unique_sensors)):
    for j in range(len(unique_sensors)):
        edge_index.append([i, j])  # Add edges between all nodes

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
print(f'edge_index : {edge_index}')

# Step 4: Define GNN Model
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# Step 5: Train the Model
model = GNN(in_channels=3, hidden_channels=8, out_channels=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index).squeeze()
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Step 6: Make Predictions
model.eval()  # Set the model to evaluation mode
predictions = model(x, edge_index).squeeze().detach().numpy()  # Get model predictions (normalized)
print("Predictions (pm2.5):", predictions)

# Separate scalers for features and target variable
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Fit the scaler on the pm2p5 target values (denormalization requires this)
y_reshaped = y.numpy().reshape(-1, 1)  # Ensure 2D for scaler compatibility
target_scaler.fit(y_reshaped)

# Predictions were normalized; now reshape for inverse_transform
predictions_reshaped = predictions.reshape(-1, 1)
denormalized_pm25 = target_scaler.inverse_transform(predictions_reshaped).flatten()

# Calculate Accuracy Metrics
actuals = y.numpy()  # Ground truth
mae = mean_absolute_error(actuals, denormalized_pm25)
mse = mean_squared_error(actuals, denormalized_pm25)
r2 = r2_score(actuals, denormalized_pm25)

# Print Results
print("Denormalized Predictions (pm2.5):", denormalized_pm25)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)
