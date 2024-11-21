import numpy as np
import matplotlib.pyplot as plt

# Define the Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generate input values
x = np.linspace(-10, 10, 100)  # 100 points from -10 to 10
print(f'x: {x}')
y = sigmoid(x)
print(f'y: {y}')

# Plot the Sigmoid function
plt.figure(figsize=(6, 4))
plt.plot(x, y, label="Sigmoid Function", color='blue')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.axhline(1, color='black', linestyle='--', linewidth=0.5)
plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
plt.title("Sigmoid Activation Function")
plt.xlabel("Input (x)")
plt.ylabel("Output (f(x))")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
