import numpy as np
from scipy import signal
from numpy.typing import NDArray

from scratchNNlib.Layers.LinearLayer import LinearLayer

class Convolution(LinearLayer):
    def __init__(self, input_shape: tuple[int], kernel_size: int, depth: int, stride: int = 1, zero_padding: int = 0):
        """Convolution Layer
        Args:
            input_shape: tuple[int] of the form (input_depth, input_height, input_width)
            kernel_size: Determines kernel dimensions as kernel_size x kernel size
            depth: The depth of the output tensor
            stride: Require this parameter to be > 0. Determines how much we slide the kernel
            zero_padding: Require this parameter to be $\geq$ 0. Determines how much padding we have around the tensor
        """

        self.only_for_training = False

        
        self.input_shape = input_shape
        self.depth = depth
        self.stride, self.zero_padding = stride, zero_padding
        self.input_depth, self.input_height, self.input_width = input_shape
        

        self.output_shape = (depth, int((self.input_height - kernel_size + (2 * zero_padding))/stride + 1), int((self.input_width - kernel_size  + (2 * zero_padding))/stride + 1))
        self.kernels_shape = (depth, self.input_depth, kernel_size, kernel_size)

        self.kernels = np.random.randn(*self.kernels_shape) * np.sqrt(2/(self.input_depth * self.input_height * self.input_width)) # http://arxiv-web3.library.cornell.edu/abs/1502.01852
        self.bias = np.random.randn(*self.output_shape)
        
        self.coefficents = [
            self.kernels,
            self.bias
        ]
        
        self.coefficents_derivatives = [
            #lambda inputs, output_gradiant: np.array([[signal.correlate(inputs[input_depth], output_gradiant[depth], mode = 'valid') for input_depth in range(self.input_depth)] for depth in range(self.depth)]),
            self.kernels_grad,
            lambda inputs, output_gradiant: output_gradiant
        ]

    
    def forward(self, inputs: NDArray) -> NDArray:
        output = self.bias.copy()
        for depth in range(self.depth):
            for input_depth in range(self.input_depth):
                output[depth] += signal.correlate2d(inputs[input_depth], self.kernels[depth, input_depth], mode = 'valid')
        return output
        
    def gradiant(self, inputs: NDArray, output_gradient: NDArray) -> NDArray:
        """
        Args:
            inputs: ndarray of shape (M x 1)
            output_gradient: ndarray of shape (A x 1)

        Returns:
            Linear output ndarray of shape (M x 1)
        """
        output = np.zeros(self.input_shape)
        for depth in range(self.depth):
            for input_depth in range(self.input_depth):
                output[input_depth] += signal.convolve2d(output_gradient[depth], self.kernels[depth, input_depth], mode = 'full')
        return output
    
    def kernels_grad(self, inputs, output_gradiant):
        kernels_grad = np.zeros(self.kernels_shape)
        for depth in range(self.depth):
            for input_depth in range(self.input_depth):
                kernels_grad[depth, input_depth] = signal.correlate2d(inputs[input_depth], output_gradiant[depth], mode = 'valid')
        return kernels_grad
    