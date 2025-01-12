import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample, find_peaks

# Original signal
f = 5  # Frequency of the signal (Hz)
t = np.linspace(-1, 1, 1000)  # Time vector
a = np.cos(2 * np.pi * f * t)  # Define the Original Signal

# Sampling frequencies
fs1 = 8  # i: fs < 2f
fs2 = 10  # ii: fs = 2f
fs3 = 50  # iii: fs >> 2f

# Sampling intervals
Ts1 = 1 / fs1
Ts2 = 1 / fs2
Ts3 = 1 / fs3

# Sampled time vectors
t1 = np.arange(-1, 1, Ts1)
t2 = np.arange(-1, 1, Ts2)
t3 = np.arange(-1, 1, Ts3)

# Sampled signals
a1 = np.cos(2 * np.pi * f * t1)
a2 = np.cos(2 * np.pi * f * t2)
a3 = np.cos(2 * np.pi * f * t3)

# Plotting

# i. Plot the original signal
plt.figure(1)
plt.plot(t, a, 'b', linewidth=1.5)
plt.title('Original Signal: a = cos(2π f t)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend(['Original Signal'])
plt.show()

# ii. Plot impulse trains for three sampling frequencies
plt.figure(2)
plt.stem(t1, np.ones_like(t1),  linefmt='r-', basefmt=" ", label="fs < 2f")
plt.stem(t2, 2 * np.ones_like(t2), linefmt='r-', basefmt=" ",  label="fs = 2f")
plt.stem(t3, 3 * np.ones_like(t3),   linefmt='r-', basefmt=" ",  label="fs >> 2f")
plt.title('Impulse Trains for Different Sampling Frequencies')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()





# iii. Plot the sampled signals with the original signal
plt.figure(3)

plt.subplot(3, 1, 1)
plt.stem(t1, a1, 'r', linefmt='r-')
plt.plot(t, a, 'b--')  # Overlay original signal
plt.title('Sampled Signal: i. (fs < 2f)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(['Sampled Signal', 'Original Signal'])
plt.grid(True)

plt.subplot(3, 1, 2)
plt.stem(t2, a2, 'g',linefmt='r-')
plt.plot(t, a, 'b--')  # Overlay original signal
plt.title('Sampled Signal: ii. (fs = 2f)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(['Sampled Signal', 'Original Signal'])
plt.grid(True)

plt.subplot(3, 1, 3)
plt.stem(t3, a3, 'm',linefmt='r-')
plt.plot(t, a, 'b--')  # Overlay original signal
plt.title('Sampled Signal: iii. (fs >> 2f)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(['Sampled Signal', 'Original Signal'])
plt.grid(True)

plt.show()

# d) Reconstruct the signal from the three sampled signals using sinc interpolation
def sinc(x):
    return np.sinc(x / np.pi)

# Reconstruct the signals using sinc interpolation
reconstructed_a1 = np.zeros_like(t)
reconstructed_a2 = np.zeros_like(t)
reconstructed_a3 = np.zeros_like(t)

for k in range(len(t1)):
    reconstructed_a1 += a1[k] * sinc(fs1 * (t - t1[k]))

for k in range(len(t2)):
    reconstructed_a2 += a2[k] * sinc(fs2 * (t - t2[k]))

for k in range(len(t3)):
    reconstructed_a3 += a3[k] * sinc(fs3 * (t - t3[k]))

# Plot the reconstructed signals along with the original signal
plt.figure()

# Case i: fs < 2f
plt.subplot(3, 1, 1)
plt.plot(t, a )
plt.plot(t, reconstructed_a1 )
plt.title('Reconstructed Signal: Case i (fs < 2f)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(['Original Signal', 'Reconstructed Signal'])
plt.grid(True)

# Case ii: fs = 2f
plt.subplot(3, 1, 2)
plt.plot(t, a )
plt.plot(t, reconstructed_a2 )
plt.title('Reconstructed Signal: Case ii (fs = 2f)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(['Original Signal', 'Reconstructed Signal'])
plt.grid(True)

# Case iii: fs >> 2f
plt.subplot(3, 1, 3)
plt.plot(t, a )
plt.plot(t, reconstructed_a3 )
plt.title('Reconstructed Signal: Case iii (fs >> 2f)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(['Original Signal', 'Reconstructed Signal'])
plt.grid(True)

plt.show()

# 2. Generate a frequency spectrum plot for:
# (a) The original signal.
# (b) The sampled signal at the three different frequencies.
# (c) The reconstructed signal.

# FFT for the reconstructed signals
A1_rec = np.fft.fftshift(np.fft.fft(reconstructed_a1))  # FFT of reconstructed signal at fs1
n1 = len(A1_rec)  # Frequency axis for the FFT
f1 = np.linspace(-fs1/2, fs1/2, n1)  # Frequency vectors

A2_rec = np.fft.fftshift(np.fft.fft(reconstructed_a2))  # FFT of reconstructed signal at fs2
n2 = len(A2_rec)  # Frequency axis for the FFT
f2 = np.linspace(-fs2/2, fs2/2, n2)  # Frequency vectors

A3_rec = np.fft.fftshift(np.fft.fft(reconstructed_a3))  # FFT of reconstructed signal at fs3
n3 = len(A3_rec)  # Frequency axis for the FFT
f3 = np.linspace(-fs3/2, fs3/2, n3)  # Frequency vectors

# (a) Original signal spectrum
plt.figure()
plt.subplot(4, 1, 1)
plt.plot(t, np.abs(a))  # Magnitude of signal
plt.title('Frequency Spectrum of Original Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)

# (b) The sampled signal at the three different frequencies
# Plot spectrum of sampled signal at fs1
plt.subplot(4, 1, 2)
plt.plot(t1, np.abs(a1))  # Magnitude of sampled signal
plt.title('Frequency Spectrum of Sampled Signal (fs = 8 Hz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)

# Plot spectrum of sampled signal at fs2
plt.subplot(4, 1, 3)
plt.plot(t2, np.abs(a2))  # Magnitude of sampled signal
plt.title('Frequency Spectrum of Sampled Signal (fs = 10 Hz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)

# Plot spectrum of sampled signal at fs3
plt.subplot(4, 1, 4)
plt.plot(t3, np.abs(a3))  # Magnitude of sampled signal
plt.title('Frequency Spectrum of Sampled Signal (fs = 50 Hz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)

plt.show()

# (c) The reconstructed signal.
plt.figure()

# Plot spectrum of reconstructed signal at fs1
plt.subplot(3, 1, 1)
plt.plot(f1, np.abs(A1_rec))  # Magnitude of FFT
plt.title('Frequency Spectrum of Reconstructed Signal (fs = 8 Hz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)

# Plot spectrum of reconstructed signal at fs2
plt.subplot(3, 1, 2)
plt.plot(f2, np.abs(A2_rec))  # Magnitude of FFT
plt.title('Frequency Spectrum of Reconstructed Signal (fs = 10 Hz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)

# Plot spectrum of reconstructed signal at fs3
plt.subplot(3, 1, 3)
plt.plot(f3, np.abs(A3_rec))  # Magnitude of FFT
plt.title('Frequency Spectrum of Reconstructed Signal (fs = 50 Hz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)

plt.show()
