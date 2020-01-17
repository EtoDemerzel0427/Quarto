import numpy as np

class Activation:
    @staticmethod
    def forward(x):
        raise NotImplementedError

    @staticmethod
    def backward(dout, cache):
        raise NotImplementedError



class Sigmoid(Activation):
    @staticmethod
    def forward(x):
        output = 1. / (1. + np.exp(-x))
        cache = x
        return output, cache

    @staticmethod
    def backward(dout, cache):
        sig = 1./ (1. + np.exp(-cache))
        grad = sig * (1 - sig)

        return dout * grad


class Relu(Activation):
    @staticmethod
    def forward(x):
        output = np.maximum(0, x)
        cache = x

        return output, cache

    def backward(dout, cache):
        grad_output = np.array(dout, copy=True)
        grad_output[cache < 0] = 0

        return grad_output

class Softmax(Activation):
    @staticmethod
    def forward(x):
        pass

    @staticmethod
    def backward(dout, cache):
        pass

