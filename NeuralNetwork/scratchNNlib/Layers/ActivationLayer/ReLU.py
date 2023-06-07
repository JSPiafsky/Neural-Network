import numpy as np
from numpy.typing import NDArray

from scratchNNlib.Layers.ActivationLayer.Activation import Activation

class ReLU(Activation):      
    def forward(self, inputs):
        return np.maximum(inputs, 0)
    
    def derivative(self, inputs):
        bool_forward = inputs > 0
        return bool_forward
        