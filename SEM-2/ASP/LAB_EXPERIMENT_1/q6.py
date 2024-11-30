import matplotlib.pyplot as plt
import numpy as np

# continuous cosine sequence
def do_cosine_sequence_continuous(length, amplitude=1, frequency=1, phase=0):
    x = np.linspace(0, 2 * np.pi, length)  # Generate evenly spaced points
    y = amplitude * np.cos(frequency * x + phase)
    return x, y

# discrete cosine sequence
def do_cosine_sequence_discrete(length, amplitude=1, frequency=1, phase=0):
    n = np.arange(length)  # Discrete time samples
    y = amplitude * np.cos(frequency * n + phase)
    return n, y

# Parameters
length = 10
amplitude = 1
frequency = 1
phase = 0

# continuous cosine sequence
x_cont, cosine_cont = do_cosine_sequence_continuous(length, amplitude, frequency, phase)

# discrete cosine sequence
n_discrete, cosine_discrete = do_cosine_sequence_discrete(length, amplitude, frequency, phase)

plt.figure(figsize=(12, 3))

# Continuous cosine wave plot (left)
plt.subplot(1, 2, 1)
plt.plot(x_cont, cosine_cont, label='Continuous Cosine Sequence', color='blue')
plt.title("Continuous Cosine Wave")
plt.xlabel("Time (t)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# Discrete cosine wave plot (right)
plt.subplot(1, 2, 2)
plt.stem(n_discrete, cosine_discrete, label='Discrete Cosine Sequence', basefmt=" ",
         linefmt='r-', markerfmt='ro')
plt.title("Discrete Cosine Wave")
plt.xlabel("Sample Index (n)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
