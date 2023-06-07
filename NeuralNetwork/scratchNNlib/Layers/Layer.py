import numpy as np
from numpy.typing import NDArray

class Layer():
    def __init__(self):
        """Base Class for a Neural Network layer"""
        self.coefficents = []
        self.coefficents_derivatives = []
        
    def __call__(self, inputs: NDArray) -> NDArray:
        """
        Args:
            inputs: ndarray of shape (M x N)

        Returns:
            Linear output ndarray of shape (A x B)
        """
        self.inputs = inputs
        return self.forward(inputs)
    
    def forward(self, inputs: NDArray) -> NDArray:
        """
        Args:
            inputs: ndarray of shape (M x N)

        Returns:
            Linear output ndarray of shape (A x B)
        """
        pass
        
    def gradiant(self, inputs: NDArray, output_gradient: NDArray) -> NDArray:
        """
        Args:
            inputs: ndarray of shape (M x N)
            output_gradient: ndarray of shape (A x B)

        Returns:
            Linear output ndarray of shape (M x N)
        """
        pass
    

    
    def backward(self, output_gradient: NDArray, learning_rate: float, inputs: NDArray = False) -> NDArray:
        """
        Args:
            output_gradient: ndarray of shape (A x B)
            learning_rate: float
            inputs: ndarray of shape (M x N), for testing class without a forward call

        Returns:
            Linear output ndarray of shape (M x N)
        """
        pass