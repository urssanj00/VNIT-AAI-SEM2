import numpy as np


# Activation function (Step function)
def step_function(x):
    return 1 if x >= 0 else 0


# Define perceptron
def perceptron(inputs, weights, bias):
    weighted_sum = np.dot(inputs, weights) + bias
    return step_function(weighted_sum)


# XNOR logic implementation
def xnor_logic(x1, x2):
    # Inputs and weights for each perceptron
    inputs = np.array([x1, x2])

    # Hidden layer
    # Perceptron 1: x1 AND NOT x2
    w1 = [1, -1]  # Weights
    b1 = -0.5  # Bias
    p1 = perceptron(inputs, w1, b1)

    # Perceptron 2: NOT x1 AND x2
    w2 = [-1, 1]
    b2 = -0.5
    p2 = perceptron(inputs, w2, b2)

    # Perceptron 3: XOR output
    w3 = [1, 1]
    b3 = -0.5
    xor = perceptron([p1, p2], w3, b3)

    # Perceptron 4: NOT XOR (XNOR output)
    w4 = [-1]
    b4 = 0.5
    xnor = perceptron([xor], w4, b4)

    return xnor


# Test the XNOR gate
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
print("XNOR Truth Table:")
for x1, x2 in inputs:
  #  print(f'{x1}, {x2}')
    print(f"Input: ({x1}, {x2}) => Output: {xnor_logic(x1, x2)}")
