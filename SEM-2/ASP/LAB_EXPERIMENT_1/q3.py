import numpy as np
import matplotlib.pyplot as plt

# discrete ramp sequence
def do_discrete_ramp_sequence(start, stop, step):
    n = np.arange(start, stop + 1, step)  # Discrete range for n
    ramp = n
    return n, ramp

# continuous ramp signal
def do_continuous_ramp_sequence(t_start, t_stop, t_step):
    t = np.arange(t_start, t_stop, t_step)  # Continuous time range
    r_continuous = t
    return t, r_continuous

# Parameters for the ramp sequences
start = 0
stop = 10
step = 2

# discrete ramp sequence
n_values, ramp_discrete = do_discrete_ramp_sequence(start, stop, step)

# continuous ramp sequence with smaller steps
t_values, ramp_continuous = do_continuous_ramp_sequence(0, stop+1, 0.1)

# Create side-by-side subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 3))

# Plot discrete ramp sequence
axes[0].stem(n_values, ramp_discrete, basefmt=" ", linefmt="red", markerfmt="o",
             label="Discrete Ramp Sequence")
axes[0].set_title("Discrete Ramp Sequence")
axes[0].set_xlabel("Index (n)")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True)
axes[0].legend()

# Plot continuous ramp sequence
axes[1].plot(t_values, ramp_continuous, color='blue',
             label="Continuous Ramp Sequence", linewidth=2)
axes[1].set_title("Continuous Ramp Sequence")
axes[1].set_xlabel("Time (t)")
axes[1].set_ylabel("Amplitude")
axes[1].grid(True)
axes[1].legend()

# Adjust layout for better spacing
plt.tight_layout()
plt.show()
