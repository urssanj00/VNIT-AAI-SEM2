import matplotlib.pyplot as plt
import numpy as np

# continuous sine sequence
def do_sine_sequence_continuous(length, amplitude=1, frequency=1, phase=0):
    x = np.linspace(0, 2 * np.pi, length)  # Generate evenly spaced points
    y = amplitude * np.sin(frequency * x + phase)
    return x, y

# discrete sine sequence
def do_sine_sequence_discrete(length, amplitude=1, frequency=1, phase=0):
    n = np.arange(length)  # Discrete time samples
    y = amplitude * np.sin(frequency * n + phase)
    return n, y

# Parameters for the sine sequences
length = 10
amplitude = 1
frequency = 1
phase = 0

# continuous sine sequence
x_cont, sine_cont = do_sine_sequence_continuous(length, amplitude, frequency, phase)

# discrete sine sequence
n_discrete, sine_discrete = do_sine_sequence_discrete(length, amplitude, frequency, phase)

plt.figure(figsize=(12, 3))

# Continuous sine wave plot (left)
plt.subplot(1, 2, 1)
plt.plot(x_cont, sine_cont, label='Continuous Sine Sequence', color='blue')
plt.title("Continuous Sine Wave")
plt.xlabel("Time (t)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# Discrete sine wave plot (right)
plt.subplot(1, 2, 2)
plt.stem(n_discrete, sine_discrete, label='Discrete Sine Sequence',
         basefmt=" ", linefmt='r-', markerfmt='ro')
plt.title("Discrete Sine Wave")
plt.xlabel("Sample Index (n)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()
