import numpy as np
class ConvolutionTest:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def signal_manual_convolution(self):
        x_len = len(self.x)
        y_len = len(self.y)
        result_len = x_len + y_len - 1
        result = [0] * result_len
        for i in range(x_len):
            for j in range(y_len):
                result[i + j] += self.x[i] * self.y[j]

        return result

    def signal_api_convolution(self):
        numpy_result = np.convolve(self.x, self.x, 'full')
        return numpy_result


# Test signals
x = [1, 1, 2, 3]
y = [0.2, 0.5, 0.3]

convolutionTest = ConvolutionTest(x, y)
manual_result = convolutionTest.signal_manual_convolution()
api_result = convolutionTest.signal_api_convolution()
print(f'Manual Convolution : {manual_result}')
print(f'API Convolution : {api_result}')