import numpy as np
import matplotlib.pyplot as plt

# discrete exponential sequence
def do_exponential_sequence_discrete(base, length):
    return [base ** n for n in range(length)]

# continuous exponential sequence
def do_exponential_sequence_continuous(base, t_start, t_stop, t_step):
    t = np.arange(t_start, t_stop, t_step)  # Continuous time range
    exp_continuous = base ** t  # Exponential function
    return t, exp_continuous

# Parameters for the exponential sequences
exp_base = 2
length = 5
t_start = 0
t_stop = 5
t_step = 0.1

# discrete exponential sequence
exp_sequence_discrete = do_exponential_sequence_discrete(exp_base, length)

# continuous exponential sequence
t_values_continuous, exp_sequence_continuous = (
    do_exponential_sequence_continuous(exp_base, t_start, t_stop, t_step))

# Create side-by-side subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 3))

# Plot the discrete exponential sequence
axes[0].stem(range(length), exp_sequence_discrete, basefmt=" ", linefmt="red",
             markerfmt="o", label="Discrete Exponential")
axes[0].set_title("Discrete Exponential Sequence")
axes[0].set_xlabel("Discrete Time")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True)
axes[0].legend()

# Plot the continuous exponential sequence
axes[1].plot(t_values_continuous, exp_sequence_continuous, color='blue',
             label="Continuous Exponential", linewidth=2)
axes[1].set_title("Continuous Exponential Sequence")
axes[1].set_xlabel("Time (t)")
axes[1].set_ylabel("Amplitude")
axes[1].grid(True)
axes[1].legend()

# Adjust layout for better spacing
plt.tight_layout()
plt.show()
