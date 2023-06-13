import numpy as np
from numpy.typing import NDArray

from scratchNNlib.Layers.LinearLayer import Layer

class Reshape(Layer):
    def __init__(self, input_shape: tuple[int], output_shape: tuple[int]):
        """Layer takes a numpy tensor of input_shape, and spits out a tensor of shape output_shape
        Requires that the product of input_shape = the product of output_shape"""
        super().__init__()

        if np.prod(input_shape) != np.prod(output_shape):
            raise ValueError('The product of the input_shape must equal the product of the output shape')
        self.input_shape = input_shape
        self.output_shape = output_shape
        
    
    def forward(self, inputs: NDArray) -> NDArray:
        """
        Args:
            inputs: ndarray of shape (M x N)

        Returns:
            Linear output ndarray of shape (A x B)
        """
        if inputs.shape != self.input_shape:
            raise ValueError(f'Expected inputs of shape {self.input_shape}, instead got input of shape {inputs.shape}')
        return np.reshape(inputs, self.output_shape)

    
    def backward(self, output_gradient: NDArray, learning_rate: float, inputs: NDArray = False) -> NDArray:
        """
        Args:
            output_gradient: ndarray of shape (A x B)
            learning_rate: float
            inputs: ndarray of shape (M x N), for testing class without a forward call

        Returns:
            Linear output ndarray of shape (M x N)
        """
        return np.reshape(output_gradient, self.input_shape)