import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First layer
        self.fc2 = nn.Linear(hidden_size, output_size) # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation to the first layer
        x = self.fc2(x)          # Output layer (without activation for regression)
        return x

input_size = 10       # Number of input features
hidden_size = 5       # Number of hidden neurons
output_size = 1       # Output size (for example, a regression task)

model = SimpleNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
print(f'criterion:{criterion}')
print(f'model.parameters():{model.parameters()}')

optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer
print(f'optimizer:{optimizer}')

