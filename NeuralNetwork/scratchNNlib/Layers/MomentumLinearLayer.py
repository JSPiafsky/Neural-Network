import numpy as np
from numpy.typing import NDArray

from scratchNNlib.Layers.LinearLayer import LinearLayer

class MomentumLinearLayer(LinearLayer):
        def __init__(self, input_size: int, output_size: int, mu: float = 0.99):
            """Implementation of Nesterov Momentum.
            Optional perimeter mu is a float specifying the 'friction' during gradient descent """
            
            super().__init__(input_size, output_size)
            self.velocity = [0, 0];
            self.last_velocity = [0, 0];
            self.mu = mu;

            self.coefficents_derivatives = [
                lambda inputs, output_gradiant: np.dot(output_gradiant, inputs.T),
                lambda inputs, output_gradiant: output_gradiant
            ]
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
            for coefficent, gradiant, velocity, last_velocity in zip(self.coefficents, gradiants, self.velocity, self.last_velocity): # Update Weight and Bias
                last_velocity = velocity
                velocity = self.mu * velocity - learning_rate * gradiant
                coefficent += -self.mu * last_velocity + (1 + self.mu) * velocity
            return self.gradiant(inputs, output_gradient) # Return Gradient