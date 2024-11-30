import matplotlib.pyplot as plt
import numpy as np

# combined sine sequence
def do_combined_sine_sequence(length, f1, f2, f3, A1, A2, A3, phi_1, phi_2, phi_3):
    x = np.linspace(0, 2 * np.pi, length)  # do evenly spaced points
    y1 = A1 * np.sin(f1 * x + phi_1)
    y2 = A2 * np.sin(f2 * x + phi_2)
    y3 = A3 * np.sin(f3 * x + phi_3)

    return x, y1, y2, y3

# Parameters
length = 500
f1, f2, f3 = 1, 2, 3
A1, A2, A3 = 1, 0.5, 0.8
phi_1, phi_2, phi_3 = 0, np.pi/4, np.pi/2

# combined sine sequence
x, y1, y2, y3 = do_combined_sine_sequence(length, f1, f2, f3, A1, A2, A3, phi_1, phi_2, phi_3)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='Sine Wave 1 (f1, A1, phi_1)', linestyle='dotted')
plt.plot(x, y2, label='Sine Wave 2 (f2, A2, phi_2)', linestyle='dashed')
plt.plot(x, y3, label='Sine Wave 3 (f3, A3, phi_3)', linestyle='dotted')


plt.xlabel("X")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.show()
