import numpy as np
from numpy.typing import NDArray

class NN:
    def __init__(self):
        """Neural Network Base Class"""
        self.chain = []

    def __call__(self, inputs: NDArray, training: bool = True) -> NDArray:
        """
        Args:
            inputs: ndarray of shape (M x N)

        Returns:
            Linear output ndarray of shape (A x B)
        """
        return self.forwardProp(inputs, training)
        
    def forwardProp(self, inputs: NDArray, training: bool = True) -> NDArray:
        """a
        Args:
            inputs: ndarray of shape (M x N)

        Returns:
            Linear output ndarray of shape (A x B)
        """
        
        output = inputs
        for layer in self.chain:
            if training or not layer.only_for_training:
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