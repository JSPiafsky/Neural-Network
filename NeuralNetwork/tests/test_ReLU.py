from scratchNNlib.Layers.ActivationLayers import ReLU
import numpy as np

a = np.array([[1,-2,3,-4]])
b = np.array([[-1,23,0]])

relu = ReLU()


def test_forward_pass():
    assert np.all(relu.forward(a) == np.array([[1,0,3, 0]]))

def test_derivative():
    assert np.all(relu.derivative(a) == np.array([[1,0,1, 0]]))
    