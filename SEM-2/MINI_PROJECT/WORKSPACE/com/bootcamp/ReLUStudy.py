import numpy as np
import matplotlib.pyplot as plt

# Define the ReLU function
def relu(x):
    return np.maximum(0, x)

# Example input data
x = np.linspace(-10, 10, 100)  # 100 points between -10 and 10
print(f'x: {x}')
y = relu(x)
print(f'y: {y}')


# Plotting the ReLU function
plt.figure(figsize=(6, 4))
plt.plot(x, y, label="ReLU Function")
plt.axhline(0, color='black', linestyle='-.', linewidth=0.3)
plt.axvline(0, color='red', linestyle=':', linewidth=0.7)
plt.title("ReLU Activation Function")
plt.xlabel("Input (x)")
plt.ylabel("Output (f(x))")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
