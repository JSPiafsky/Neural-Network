import numpy as np
from numpy.typing import NDArray

from scratchNNlib.Loss.Loss import Loss

class RootMeanSquare(Loss):
    def forward(self, network_output, actual_value):
        super().forward(network_output, actual_value)
        rootError = np.sum((actual_value - network_output)**2)
        return np.sqrt(rootError/len(network_output))
    
    def gradiant(self, network_output, actual_value, input_length):
        grad = (1/self.forward(network_output, actual_value)) * (network_output - actual_value)/input_length
        return grad
        