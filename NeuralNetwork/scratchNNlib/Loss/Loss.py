import numpy as np
from numpy.typing import NDArray

class Loss():
    
    def forward(self, network_output: NDArray, actual_value: NDArray) -> float:
        """
        Args:
            network_output: ndarray of shape (M x N)
            actual_value: ndarray of shape (M x N)

        Returns:
            float
        """
        self.network_output = network_output
        self.actual_value = actual_value
    
    def __call__(self, network_output: NDArray, actual_value: NDArray) -> float:
        """
        Args:
            network_output: ndarray of shape (M x N)
            actual_value: ndarray of shape (M x N)

        Returns:
            float
        """
        return self.forward(network_output, actual_value)
    
    def gradiant(self, network_output: NDArray, actual_value: NDArray, input_length: int) -> NDArray:
        """
        Args:
            network_output: ndarray of shape (M x N)
            actual_value: ndarray of shape (M x N)
            input_length: int

        Returns:
            ndarry of shape (M x N)
        """
        pass
    
    def backward(self):
        return self.gradiant(self.network_output, self.actual_value, np.size(self.network_output))