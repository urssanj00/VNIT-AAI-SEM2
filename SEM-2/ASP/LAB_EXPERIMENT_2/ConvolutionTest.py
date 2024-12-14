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

        print(f"signal_manual_convolution:  -> {result }")
        return result

    def signal_api_convolution(self):
        numpy_result = np.convolve(self.x, self.y, 'full')
        return numpy_result

    def wave_signal_api_convolution(self):
        convolve_result = signal_convolve(self.x, self.y, 'full')
        return convolve_result

    def plot_helper(self, signal_data, discrete, sub_plot, p_label, p_marker):
        # Set bar width for thin bars
        bar_width = 0.03

        # Create a time range for plotting
        time_signal = range(len(signal_data))

        if discrete:
            sub_plot.bar(time_signal, signal_data, width=bar_width, label=p_label)
        else:
            sub_plot.plot(time_signal, signal_data, label=p_label, marker=p_marker)

        sub_plot.legend()
        return sub_plot



    def plot_convolved_signal(self, manual_convolved_signal, api_convolved_signal, plot_file_name,
                              x_label, y_label, discrete=True):

        # Define the signals and convolution results
        signal_x = self.x
        signal_y = self.y
        manual_convolution_1_of_x_and_y = manual_convolved_signal
        api_convolution_2_of_x_and_y = api_convolved_signal

        # Plotting
        figure = plt.figure(figsize=(13, 8))
        grid_spec = gridspec.GridSpec(3, 2, width_ratios=[1, 1])

        # Plot signal_x
        subplot_x = plt.subplot(grid_spec[0, 0])
        subplot_x = self.plot_helper(signal_x, discrete, subplot_x, x_label, 'o')

        # Plot signal_y
        subplot_y = plt.subplot(grid_spec[0, 1])
        subplot_y = self.plot_helper(signal_y, discrete, subplot_y, y_label, 'x')

        # Plot manual convolution
        subplot_manual = plt.subplot(grid_spec[1, :])
        subplot_manual = self.plot_helper(manual_convolution_1_of_x_and_y, discrete,
                                          subplot_manual,'Manual Convolution', 's')

        # Plot manual convolution
        subplot_api = plt.subplot(grid_spec[2, :])
        subplot_api = self.plot_helper(api_convolution_2_of_x_and_y, discrete,
                                       subplot_api, 'Built-in Convolution', 'd')

        plt.tight_layout()
        plt.savefig(plot_file_name)
        plt.close()


# (a) x1[n] = [1, 1, 2, 3] and x2[n] = [0.2, 0.5, 0.3]
x = [1, 1, 2, 3]
y = [0.2, 0.5, 0.3]

convolutionTest_a = ConvolutionTest(x, y)
manual_result = convolutionTest_a.signal_manual_convolution()
api_result = convolutionTest_a.signal_api_convolution()

print(f'Manual Convolution : {manual_result}')
print(f'Built-in Convolution : {api_result.tolist()}')

convolutionTest_a.plot_convolved_signal(manual_result, api_result, 'plot_q_a.png',
                                        'x1[n] = [1, 1, 2, 3]',
                                        'x2[n] = [0.2, 0.5, 0.3]', True)

# (b) x1(t) = sin(t) and x2(t) = cos(t)

# Create a time range
t = np.linspace(-2 * np.pi, 2 * np.pi, 500)
x = np.sin(t)
y = np.cos(t)

convolutionTest_b = ConvolutionTest(x, y)
manual_result = convolutionTest_b.signal_manual_convolution()
api_result = convolutionTest_b.wave_signal_api_convolution()

print(f'Manual Convolution : {manual_result}')
print(f'Built-in Convolution : {api_result.tolist()}')

convolutionTest_b.plot_convolved_signal(manual_result, api_result, 'plot_q_b.png',
                                        'x1(t) = sin(t)',
                                        'x2(t) = cos(t)', False)