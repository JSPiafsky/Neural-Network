import numpy as np
from numpy.typing import NDArray

from scratchNNlib.Loss.Loss import Loss

class BinaryCrossEntropy(Loss):
    def forward(self, network_output, actual_value):
        super().forward(network_output, actual_value)
        return (-1/network_output.size) * np.sum(np.multiply(actual_value, np.log(np.abs(network_output)) + np.multiply(1 - actual_value, np.log(1-network_output))))
    
    def gradiant(self, network_output, actual_value, input_length):
        grad = (1/input_length) * ((1 - actual_value)/(1- network_output) - (actual_value/network_output))
        return grad
        