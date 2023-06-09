from scratchNNlib.Loss import CrossEntropy
import numpy as np

net = np.array([[1,2,3,4]])
exp = np.array([[1,0,0, 0]])

loss = CrossEntropy()


def test_forward_pass():
    assert np.all(loss(net, exp)== np.array([[0]]))


def test_gradiant():
    assert np.all(relu.derivative(a) == np.array([[1,0,1, 0]]))
    