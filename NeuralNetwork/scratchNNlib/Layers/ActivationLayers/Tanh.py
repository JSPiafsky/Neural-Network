import numpy as np
from numpy.typing import NDArray

from scratchNNlib.Layers.ActivationLayers.Activation import Activation

class Tanh(Activation):
        
    def forward(self, inputs):
        return np.tanh(inputs)
    
    def derivative(self, inputs):
        grad = 1 - (np.tanh(inputs) ** 2)
        return grad