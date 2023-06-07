import numpy as np
from numpy.typing import NDArray

from scratchNNlib.Layers.Layer import Layer

class Activation(Layer):
    def derivative(self, inputs: NDArray) -> NDArray:
        pass
    
    def backward(self, output_gradient, learning_rate, inputs = False):
        inputs = self.inputs if not inputs else inputs
        return np.multiply(output_gradient, self.derivative(inputs))