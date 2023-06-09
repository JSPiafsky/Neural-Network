import numpy as np
from numpy.typing import NDArray

from scratchNNlib.Layers.ActivationLayer.Activation import Activation


class Softmax(Activation):
    
    def forward(self, inputs):
        # https://cs231n.github.io/linear-classify/#softmax
        shiftingConstant = -np.max(inputs)
        return np.exp(inputs + shiftingConstant)/np.sum(np.exp(inputs + shiftingConstant))
    
    def derivative(self, inputs):
        grad = np.multiply(self.forward(inputs), (1 - self.forward(inputs)))
        return grad
    
