import numpy as np
from numpy.typing import NDArray

from scratchNNlib.Layers.Layer import Layer

class LinearLayer(Layer):
    def __init__(self, input_size: int, output_size: int):
        """Linear Layer for Neural Network"""
        
        super().__init__()
        self.input_size, self.output_size = input_size, output_size
        self.weights = np.random.randn(self.output_size, self.input_size) * np.sqrt(2/self.input_size) # http://arxiv-web3.library.cornell.edu/abs/1502.01852
        self.bias = np.random.randn(self.output_size, 1)
        
        self.coefficents = [
            self.weights,
            self.bias
        ]
        
        self.coefficents_derivatives = [
            lambda inputs, output_gradiant: np.dot(output_gradiant, inputs.T),
            lambda inputs, output_gradiant: output_gradiant
        ]

    
    def forward(self, inputs: NDArray) -> NDArray:
        """
        Args:
            inputs: ndarray of shape (M x 1)

        Returns:
            Linear output ndarray of shape (A x 1)
        """
        if inputs.shape != (self.input_size, 1):
            raise ValueError(f'This layer expects inputs of shape {(self.input_size, 1)}, instead got input of shape {inputs.shape}')
            
        return np.dot(self.weights, inputs) + self.bias
        
    def gradiant(self, inputs: NDArray, output_gradient: NDArray) -> NDArray:
        """
        Args:
            inputs: ndarray of shape (M x 1)
            output_gradient: ndarray of shape (A x 1)

        Returns:
            Linear output ndarray of shape (M x 1)
        """
        
        return np.dot(self.weights.T, output_gradient)
    
    def backward(self, output_gradient: NDArray, learning_rate: float, inputs: NDArray = False) -> NDArray:
        """
        Args:
            output_gradient: ndarray of shape (A x 1)
            learning_rate: float
            inputs: ndarray of shape (M x 1), for testing class without a forward call

        Returns:
            Linear output ndarray of shape (M x 1)
        """
        
        inputs = self.inputs if not inputs else inputs
        gradiants = [derivative(inputs, output_gradient) for derivative in self.coefficents_derivatives] # Calculate Partials with respect to Weights and Bias
        for coefficent, gradiant in zip(self.coefficents, gradiants): # Update Weight and Bias
            coefficent -= gradiant * learning_rate
        return self.gradiant(inputs, output_gradient) # Return Gradient