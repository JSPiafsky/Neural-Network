import numpy as np
from numpy.typing import NDArray

from scratchNNlib.Layers.ActivationLayers.Activation import Activation

class ReLU(Activation):      
    def forward(self, inputs: NDArray) -> NDArray:
        return np.maximum(inputs, 0)
    
    def derivative(self, inputs: NDArray) -> NDArray:
        bool_forward = inputs > 0
        return bool_forward
        