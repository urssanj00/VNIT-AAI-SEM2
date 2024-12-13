import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve as signal_convolve
import matplotlib.gridspec as gridspec
from setuptools.msvc import winreg


class ConvolutionTest:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def signal_manual_convolution(self):
        x_len = len(self.x)
        y_len = len(self.y)

        result_len = x_len + y_len - 1
        # make a zero initialize array to contain the product of two matrices x and y
        result = [0] * result_len

        for i in range(x_len):
            for j in range(y_len):
                result[i + j] += self.x[i] * self.y[j]
                print(f"signal_manual_convolution: result[{i + j}] += {self.x[i]} * {self.x[j]} -> {result[i + j]}")

        return result

    def signal_api_convolution(self):
        numpy_result = np.convolve(self.x, self.y, 'full')
        return numpy_result

    def signal_api_convolution(self):
        convolve_result = signal_convolve(self.x, self.y, 'full')
        return convolve_result

    def plot_convolved_signal(self, manual_convolved_signal, api_convolved_signal, plot_file_name):

        # Define the signals and convolution results
        signal_x = self.x
        signal_y = self.y
        manual_convolution_1_of_x_and_y = manual_convolved_signal
        api_convolution_2_of_x_and_y = api_convolved_signal

        # Create a time range for plotting
        time_signal = range(len(signal_x))
        time_convolution = range(len(manual_convolution_1_of_x_and_y))

        # Plotting
        figure = plt.figure(figsize=(13, 8))
        grid_spec = gridspec.GridSpec(3, 2, width_ratios=[1, 1])

        # Plot signal_x
        subplot_x = plt.subplot(grid_spec[0, 0])
        subplot_x.plot(time_signal, signal_x, label='Signal X', marker='o')
        subplot_x.legend()

        # Plot signal_y
        subplot_y = plt.subplot(grid_spec[0, 1])
        subplot_y.plot(time_signal, signal_y, label='Signal Y', marker='x')
        subplot_y.legend()

        # Plot manual convolution
        subplot_manual = plt.subplot(grid_spec[1, :])
        subplot_manual.plot(time_convolution, manual_convolution_1_of_x_and_y, label='Manual Convolution', marker='s')
        subplot_manual.legend()

        # Plot manual convolution
        subplot_api = plt.subplot(grid_spec[2, :])
        subplot_api.plot(time_convolution, api_convolution_2_of_x_and_y, label='API Convolution', marker='d')
        subplot_api.legend()

        plt.tight_layout()
        plt.savefig(plot_file_name)
        plt.close()
        # Display the plot
        #plt.show()


# (a) x1[n] = [1, 1, 2, 3] and x2[n] = [0.2, 0.5, 0.3]
x = [1, 2, 3]
y = [0.2, 0.5, 0.3]

convolutionTest_a = ConvolutionTest(x, y)
manual_result = convolutionTest_a.signal_manual_convolution()
api_result = convolutionTest_a.signal_api_convolution()

print(f'Manual Convolution : {manual_result}')
print(f'API Convolution : {api_result.tolist()}')

convolutionTest_a.plot_convolved_signal(manual_result, api_result, 'plot_q_a.png')

# (b) x1(t) = sin(t) and x2(t) = cos(t)

# Create a time range
t = np.linspace(-2 * np.pi, 2 * np.pi, 500)
x = np.sin(t)
y = np.cos(t)

convolutionTest_b = ConvolutionTest(x, y)
manual_result = convolutionTest_b.signal_manual_convolution()
api_result = convolutionTest_b.signal_api_convolution()

print(f'Manual Convolution : {manual_result}')
print(f'API Convolution : {api_result.tolist()}')

convolutionTest_b.plot_convolved_signal(manual_result, api_result, 'plot_q_b.png')