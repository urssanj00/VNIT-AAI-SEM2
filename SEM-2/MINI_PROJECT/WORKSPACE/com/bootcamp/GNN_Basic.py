import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch.nn import Linear

# Define the dataset
node_features = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                              [5.0, 6.0, 7.0, 8.0],
                              [9.0, 10.0, 11.0, 12.0],
                              [13.0, 14.0, 15.0, 16.0]], dtype=torch.float)
node_targets = torch.tensor([1.5, 2.0, 3.5, 4.0], dtype=torch.float)
edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

# Create the graph data object
data = Data(x=node_features, y=node_targets, edge_index=edge_index)


# Define the GNN model
class GNNModel(torch.nn.Module):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(4, 16)  # 4 input features, 16 output features
        self.conv2 = GCNConv(16, 8)  # 16 input features, 8 output features
        self.fc = Linear(8, 1)  # Final regression layer

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)  # Regression output
        return x.squeeze()  # Ensure shape matches target


# Initialize model, optimizer, and loss function
model = GNNModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()  # Clear gradients

    # Forward pass
    output = model(data)

    # Compute loss
    loss = criterion(output, data.y)

    # Backward pass
    loss.backward()
    optimizer.step()

    # Logging
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(data)
    print("\nPredictions:", predictions)
    print("True Values:", data.y)
