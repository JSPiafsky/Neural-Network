import numpy as np
from numpy.typing import NDArray

from scratchNNlib.Loss.Loss import Loss

class MeanSquare(Loss):
    def forward(self, network_output, actual_value):
        super().forward(network_output, actual_value)
        rootError = np.mean(np.power(actual_value - network_output, 2))
        return rootError
    
    def gradiant(self, network_output, actual_value, input_length):
        grad = 2*(network_output - actual_value)/input_length
        return grad