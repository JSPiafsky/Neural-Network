import numpy as np
from numpy.typing import NDArray

from scratchNNlib.Layers.ActivationLayer.Activation import Activation


class Sigmoid(Activation):
    def forward(self, inputs):
        return 1/(1 + np.exp(-inputs))
    
    def derivative(self, inputs):
        grad = self.forward(inputs) * (1 - self.forward(inputs))
        return grad
    
