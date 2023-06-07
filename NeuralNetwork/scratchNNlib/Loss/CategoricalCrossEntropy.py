import numpy as np
from numpy.typing import NDArray

from scratchNNlib.Loss.Loss import Loss

class CategoricalCrossEntropy(Loss):
    def forward(self, network_output, actual_value):
        super().forward(network_output, actual_value)
        return -1*np.sum([expected_value * np.log(network_value) for expected_value, network_value in zip(network_output, actual_value)])
        #return -np.sum(actual_value * np.log(np.abs(network_output)))
    
    def gradiant(self, network_output, actual_value, input_length):
        grad = -1*np.sum([expected_value/np.abs(network_value) for expected_value, network_value in zip(network_output, actual_value)])
        return grad
    
    '''def gradiant(self, y_hat, y, T):
        grad = -1*self.actual_value/self.network_output
        return grad.T'''
        