import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    # input layer (4 features of the flower) -->
    #   a) sepal length / width
    #   b) petal length / width
    # Hidden Layer1 (number of neurons)
    # H2 (n) -->
    # O/p (3 classes of Iris flowers)
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)