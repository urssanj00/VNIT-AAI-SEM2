import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Step 1: Define a simple graph
# A graph with 4 nodes and 4 edges
# Edge list (source, target): [(0, 1), (1, 2), (2, 3), (3, 0)]
edge_index = torch.tensor([
    [0, 1, 2, 3],  # Source nodes
    [1, 2, 3, 0]   # Target nodes
], dtype=torch.long)

# Node features: Each node has a 3-dimensional feature vector
x = torch.tensor([
    [1, 0, 0],  # Node 0
    [0, 1, 0],  # Node 1
    [0, 0, 1],  # Node 2
    [1, 1, 0]   # Node 3
], dtype=torch.float)

# Labels for node classification (4 nodes, 2 classes: 0 or 1)
y = torch.tensor([0, 1, 0, 1], dtype=torch.long)

# Create the PyTorch Geometric data object
data = Data(x=x, edge_index=edge_index, y=y)
print(f'data : {data}')


# Step 2: Define a simple GNN model
class SimpleGNN(torch.nn.Module):
    def __init__(self):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(in_channels=3, out_channels=4)  # Input: 3 features, Output: 4 hidden features
        self.conv2 = GCNConv(in_channels=4, out_channels=2)  # Output: 2 classes 0, 1

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)  # First GCN layer
        x = F.relu(x)                  # Apply ReLU non-linearity
        x = self.conv2(x, edge_index)  # Second GCN layer
        return F.log_softmax(x, dim=1) # Log softmax for node classification

# Step 3: Train the GNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleGNN().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(50):  # 20 epochs of training
    model.train()
    optimizer.zero_grad()
    out = model(data)               # Forward pass
    loss = F.nll_loss(out, data.y)  # Negative log-likelihood loss
    loss.backward()                 # Backpropagation
    optimizer.step()                # Optimizer step
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# Step 4: Evaluate the model
model.eval()
_, pred = model(data).max(dim=1)  # Get predictions
correct = int((pred == data.y).sum())  # Count correct predictions
accuracy = correct / data.y.size(0)   # Calculate accuracy
print(f'Accuracy: {accuracy:.4f}')
print(f'prediction : {pred}')
