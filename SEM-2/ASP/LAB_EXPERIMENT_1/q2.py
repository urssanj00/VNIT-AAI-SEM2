import matplotlib.pyplot as plt
import numpy as np

# Function to create a unit step sequence (discrete)
def do_unit_step_sequence(range_n, shift_N):
    step_u_n = np.where(range_n >= 0, 1, 0)  # Step sequence u(n)
    step_u_n_minus_N = np.where(range_n >= shift_N, 1, 0)  # Step sequence u(n-N)
    return step_u_n - step_u_n_minus_N  # Difference: [u(n) - u(n-N)]

# Function to create a continuous unit step sequence (Heaviside function)
def continuous_unit_step(t, shift_N):
    return np.where(t >= shift_N, 1, 0)  # Heaviside step function u(t-N)

# Parameters for the sequence
shift_N = 5  # Shift length
range_n = np.arange(-5, 10)  # Discrete range of n
t_values = np.linspace(-10, 10, 1000)  # Continuous time range

# Generate the discrete unit step signal
step_sequence = do_unit_step_sequence(range_n, shift_N)

# Generate the continuous unit step signal
continuous_step_sequence = continuous_unit_step(t_values, shift_N)

# Plotting the discrete and continuous unit step sequences
plt.figure(figsize=(12, 6))

# Discrete unit step sequence plot
plt.subplot(1, 2, 1)
plt.stem(range_n, step_sequence, basefmt=" ")
plt.title("Discrete Unit Step Sequence: [u(n) - u(n - N)]")
plt.xlabel("Index (n)")
plt.ylabel("Amplitude")
plt.grid(True)

# Continuous unit step sequence plot
plt.subplot(1, 2, 2)
plt.plot(t_values, continuous_step_sequence, color='orange', label="Continuous Step")
plt.title("Continuous Unit Step Sequence (Heaviside Function)")
plt.xlabel("Time (t)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()
