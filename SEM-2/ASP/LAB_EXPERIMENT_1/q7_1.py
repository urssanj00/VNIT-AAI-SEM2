import matplotlib.pyplot as plt
import numpy as np

# Function to generate a combined sine sequence
def generate_combined_sine_sequence(length, f1, f2, f3, A1, A2, A3, phi1, phi2, phi3):
    x = np.linspace(0, 2 * np.pi, length)  # Generate evenly spaced points
    y1 = A1 * np.sin(f1 * x + phi1)
    y2 = A2 * np.sin(f2 * x + phi2)
    y3 = A3 * np.sin(f3 * x + phi3)

    return x, y1, y2, y3

# Parameters for the sine waves
length = 500  # Number of points in the sequence
f1, f2, f3 = 1, 2, 3  # Frequencies of the sine waves
A1, A2, A3 = 1, 0.5, 0.8  # Amplitudes of the sine waves
phi1, phi2, phi3 = 0, np.pi/4, np.pi/2  # Phase shifts of the sine waves

# Generate the combined sine sequence
x, y1, y2, y3 = generate_combined_sine_sequence(length, f1, f2, f3, A1, A2, A3, phi1, phi2, phi3)

# Plot the individual sine waves and the combined sequence
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label=f'Sine Wave 1 (f1={f1}, A1={A1}, φ1={phi1})', linestyle='dashed')
plt.plot(x, y2, label=f'Sine Wave 2 (f2={f2}, A2={A2}, φ2={phi2})', linestyle='dashed')
plt.plot(x, y3, label=f'Sine Wave 3 (f3={f3}, A3={A3}, φ3={phi3})', linestyle='dashed')

plt.xlabel("X (Time)")
plt.ylabel("Amplitude")
plt.title("Combined Sine Sequence with Three Frequencies, Amplitudes, and Phases")
plt.grid(True)
plt.legend()
plt.show()
