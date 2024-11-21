import torch
import torch.nn as nn


# Define a custom model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()  # Initialize nn.Module
        self.fc = nn.Linear(10, 1)      # Define a fully connected layer

    def forward(self, x):
        return self.fc(x)              # Forward pass

# Create an instance of the model
model = MyModel()

# Check trainable parameters
i = 0
for param in model.parameters():
    print(f'{i}. param.shape : {param.shape}')
    print(f'{i}. param : {param}')
    i = i+1
