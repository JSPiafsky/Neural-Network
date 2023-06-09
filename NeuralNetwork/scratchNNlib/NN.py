import numpy as np
from numpy.typing import NDArray

class NN:
    def __init__(self):
        """Neural Network Base Class"""
        self.chain = []
        
    def forwardProp(self, inputs: NDArray) -> NDArray:
        """a
        Args:
            inputs: ndarray of shape (M x N)

        Returns:
            Linear output ndarray of shape (A x B)
        """
        
        output = inputs
        for layer in self.chain:
            output = layer(output)
        return output
    
    def backProp(self, output_gradient: NDArray, learning_rate: float) -> None:
        """
        Args:
            output_gradient: ndarray of shape (A x B)
            learning_rate: float
        """
        gradiant = output_gradient
        for layer in reversed(self.chain):
            gradiant = layer.backward(gradiant, learning_rate)

        def fit(self):
            '''todo'''
            pass