# Import PyTorch
# torch: PyTorch's main library for building and working with tensors, which are the building blocks for neural networks.
# torch.nn: A submodule in PyTorch that contains classes and functions to build neural network layers and
# models (e.g., nn.Linear for fully connected layers)

import torch
import torch.nn as nn

# Define a simple binary classifier
# class BinaryClassifier(nn.Module): Defines a custom neural network model by inheriting from PyTorch's nn.Module,
# which provides the basic structure for all neural network models.
class BinaryClassifier(nn.Module):
    # __init__(): The constructor method initializes the layers and other components of the model.
    def __init__(self, input_dim):
        # super(BinaryClassifier, self).__init__(): Calls the parent class (nn.Module) constructor to ensure the model
        # is properly initialized.
        super(BinaryClassifier, self).__init__()
        # Creates a fully connected (linear) layer with input_dim inputs and 1 output neuron.
        # In binary classification, the single output represents the raw score (logit), which will later be passed
        # through the Sigmoid function.
        self.fc = nn.Linear(input_dim, 1)  # Single output neuron

    # forward(self, x): Defines how the data flows through the model during the forward pass (from input to output).
    def forward(self, x):
        # Takes the input x (a tensor of features) and applies the linear layer self.fc to compute the raw score
        # z, which is the weighted sum of inputs plus bias:  z=W⋅x+b
        # Here, W is the weight matrix, x is the input, and b is the bias.
        z = self.fc(x)
        print(z)
        print(f'z:{z}')
        # Applies the Sigmoid activation function to the raw score z, squashing it into the range [0, 1].
        # This represents the probability of the input belonging to the positive class.
        # Apply Sigmoid activation
        sigmoid_z = torch.sigmoid(z)
        print(f'sigmoid_z:{sigmoid_z}')
        return sigmoid_z

# Example input
# Generates a tensor with random values sampled from a normal distribution.
# Shape (5, 3) means 5 samples, each with 3 features.
#   Example: 5 X 3 Matrix
x = torch.randn(5, 3)  # 5 samples, each with 3 features

# Creates an instance of the BinaryClassifier class with input_dim=3, which matches the number of features in the
# input tensor x.
# The model now expects inputs with 3 features.
model = BinaryClassifier(3)

# Calls the forward method of the BinaryClassifier to compute the output.
# Inside the forward method:
#   1. Raw score (z) is computed using the linear layer: 𝑧 = 𝑊⋅𝑥 + 𝑏
#   2. Sigmoid activation is applied to z: f(x) = 1/(1+e^-x)
# The result, output, is a tensor with 5 probabilities (one for each sample).
output = model(x)

print("Raw Output (z):", model.fc(x).detach().numpy())
print("Sigmoid Output (f(x)):", output.detach().numpy())
