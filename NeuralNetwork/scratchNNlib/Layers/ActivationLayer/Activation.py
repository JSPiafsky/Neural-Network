import numpy as np
from numpy.typing import NDArray

from scratchNNlib.Layers.Layer import Layer

class Activation(Layer):
    def derivative(self, inputs: NDArray) -> NDArray:
        """
        Args:
            inputs: ndarray of shape (M x N)

        Returns:
            Linear output ndarray of shape (M x N)
        """
        pass
    
    def backward(self, output_gradient: NDArray, learning_rate: float, inputs: NDArray = False) -> NDArray:
        """
        Args:
            output_gradient: ndarray of shape(M x N)
            learning_rate: float
            inputs: ndarray of shape (M x N)

        Returns:
            Linear output ndarray of shape (M x N)
        """
        inputs = self.inputs if not inputs else inputs
        return np.multiply(output_gradient, self.derivative(inputs))