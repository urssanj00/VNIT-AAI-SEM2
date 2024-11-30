import matplotlib.pyplot as plt
import numpy as np

# Function to create a unit step sequence
def do_unit_step_sequence(range_n, shift_N):
    step_u_n = np.where(range_n >= 0, 1, 0)  # Step sequence u(n)
    step_u_n_minus_N = np.where(range_n >= shift_N, 1, 0)  # Step sequence u(n-N)
    return step_u_n - step_u_n_minus_N  # Difference: [u(n) - u(n-N)]

# Parameters for the sequence
shift_N = 5  # Shift length
range_n = np.arange(-5, 10)  # Values of n

# Generate the unit step signal
step_sequence = do_unit_step_sequence(range_n, shift_N)

# Plotting the generated unit step sequence
plt.figure(figsize=(12, 4))
plt.stem(range_n, step_sequence, basefmt=" ")
plt.title("Unit Step Sequence: [u(n) - u(n - N)]")
plt.xlabel("Index (n)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
