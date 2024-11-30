import numpy as np
import matplotlib.pyplot as plt

# discrete signal: n, spanning from -25 to 25 (inclusive) for
n_values = np.arange(-25, 26)

# discrete impulse: Initialize the unit impulse signal as an array of zeros
unit_impulse = np.zeros_like(n_values)
# Setting the impulse at the zero index
unit_impulse[n_values == 0] = 1

# Continuous impulse (Gaussian approximation of Dirac delta)
t_values = np.linspace(-5, 5, 1000)  # Create time axis for continuous signal
sigma = 0.1  # Standard deviation
continuous_impulse = np.exp(-0.5 * (t_values / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

# Plotting both the discrete and continuous impulse signals

# Plot the discrete unit impulse signal
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.stem(n_values, unit_impulse, basefmt=" ")
plt.title('Discrete Unit Impulse Signal')
plt.xlabel('Time (n)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot the continuous Gaussian approximation of impulse
plt.subplot(1, 2, 2)
plt.plot(t_values, continuous_impulse, color='orange', label="Gaussian Approximation")
plt.title('Continuous Unit Impulse (Gaussian Approximation)')
plt.xlabel('Time (t)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
