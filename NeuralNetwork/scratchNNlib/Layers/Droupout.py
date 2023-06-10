import numpy as np
from numpy.typing import NDArray

from scratchNNlib.Layers.Layer import Layer

class Dropout(Layer):
    def __init__(self, dropout_percent: float):
        """A dropout layer as described in http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf"""
        
        super().__init__()
        self.dropout_percent = dropout_percent
        self.rng = np.random.default_rng()

        self.last_architecture = 0
        self.only_for_training = True
    
    def forward(self, inputs: NDArray) -> NDArray:
        """
        Args:
            inputs: ndarray of shape (M x 1)

        Returns:
            Linear output ndarray of shape (A x 1)
        """
        self.last_architecture = self.rng.binomial(1, 1 - self.dropout_percent, size = (inputs.size, 1)) 
        return inputs * self.last_architecture * 1/(1-self.dropout_percent)
        
    def gradiant(self, inputs: NDArray, output_gradient: NDArray) -> NDArray:
        """
        Args:
            inputs: ndarray of shape (M x 1)
            output_gradient: ndarray of shape (A x 1)

        Returns:
            Linear output ndarray of shape (M x 1)
        """
        
        return output_gradient * self.last_architecture ## Figure out if we apply scaling from 1/(1-self.dropout_percent) to gradient
    
    def backward(self, output_gradient: NDArray, learning_rate: float, inputs: NDArray = False) -> NDArray:
        """
        Args:
            output_gradient: ndarray of shape (A x 1)
            learning_rate: float
            inputs: ndarray of shape (M x 1), for testing class without a forward call

        Returns:
            Linear output ndarray of shape (M x 1)
        """
        
        return self.gradiant(inputs, output_gradient) # Return Gradient