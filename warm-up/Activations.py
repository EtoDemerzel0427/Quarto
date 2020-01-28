import numpy as np

class Activation:
    @staticmethod
    def forward(x):
        raise NotImplementedError

    @staticmethod
    def backward(dout, cache):
        raise NotImplementedError



class Sigmoid(Activation):
    """
    In the forward pass, sigmoid(x) = 1 / (1 + exp(-x))
    In the backward pass, sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    """
    @staticmethod
    def forward(x):
        output = 1. / (1. + np.exp(-x))
        cache = output
        return output, cache

    @staticmethod
    def backward(dout, cache):
        grad = cache * (1 - cache)

        return dout * grad


class Relu(Activation):
    @staticmethod
    def forward(x):
        output = np.maximum(0, x)
        cache = x

        return output, cache

    @staticmethod
    def backward(dout, cache):
        grads = np.array(dout, copy=True)
        grads[cache < 0] = 0

        return grads

class Softmax(Activation):
    """
    Since CrossEntropyLoss is easier and efficient to implement,
    I will not implement a single Softmax class this time.
    """
    @staticmethod
    def forward(x):
        raise NotImplementedError

    @staticmethod
    def backward(dout, cache):
        raise NotImplementedError


if __name__ == '__main__':
    x = np.array([0.4, 1, -20, 48])

    # sigmoid test
    y, cache= Sigmoid.forward(x)
    grads = Sigmoid.backward(1, cache)

    f = lambda x: 1. / (1. + np.exp(-x))
    g = lambda x: (f(x + 0.0001) - f(x - 0.0001)) / 0.0002

    print(f(x), g(x))
    print(y, grads)



