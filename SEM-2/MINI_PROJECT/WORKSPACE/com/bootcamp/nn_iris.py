import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    # input layer (4 features of the flower) -->
    #   a) sepal length / width
    #   b) petal length / width
    # Hidden Layer1 (number of neurons) h1
    # H2 (n) -->                        h2
    # O/p (3 classes of Iris flowers)
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__() #instantiate nn.Module
        self.fc1 = nn.Linear(in_features, h1)  # in_features moves to h1
        print(f'self.fc1 : {self.fc1}')
        self.fc2 = nn.Linear(h1, h2)           # h1 (hidden layer1) moves to h1 (hidden layer2)
        print(f'self.fc2 : {self.fc2}')
        self.out = nn.Linear(h2, out_features) # h2 (hidden later2) moves to out_features
        print(f'self.out : {self.out}')


    def forward(self, x):
        print(f'0. x : {x}')
        x = F.relu(self.fc1(x))  # relu zeros negative and returns same for positives max(0, x)
        print(f'1. x : {x}')
        x = F.relu(self.fc2(x))
        print(f'2. x : {x}')
        x = self.out(x)
        print(f'3. x : {x}')

        return x

# pick a manual seed for randonization
torch.manual_seed(41)

model = Model()

# Create a sample input tensor (batch size of 1, 4 features)
input_data = torch.rand(1, 4)

# Pass the input through the model
output = model(input_data)

print(f'Output: {output}')