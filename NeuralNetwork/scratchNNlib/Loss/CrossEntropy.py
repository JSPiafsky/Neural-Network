import numpy as np
from numpy.typing import NDArray

from scratchNNlib.Loss.Loss import Loss

class CrossEntropy(Loss):
    '''Not Working ATM???'''
    def forward(self, network_output, actual_value):
        super().forward(network_output, actual_value)
        return -np.sum(np.multiply(actual_value, np.log(np.abs(network_output))))
    
    def gradiant(self, network_output, actual_value, input_length):
        grad = - actual_value/(network_output * input_length)
        return grad
        