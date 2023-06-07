import numpy as np
from numpy.typing import NDArray

class Loss():
    
    def forward(self, network_output, actual_value):
        self.network_output = network_output
        self.actual_value = actual_value
    
    def __call__(self, network_output, actual_value):
        return self.forward(network_output, actual_value)
    
    def gradiant(self, network_output, actual_value, input_length):
        pass
    
    def backward(self):
        return self.gradiant(self.network_output, self.actual_value, np.size(self.network_output))