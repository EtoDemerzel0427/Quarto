import numpy as np
from Initializers import xavier_initializer, zero_initializer
from Activations import Sigmoid, Relu, Softmax


class Module:
    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def init_parameters(self):
        raise NotImplementedError

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class Dense(Module):
    def __init__(self, input_dim: int, output_dim: int, use_bias: bool, activation: str) -> None:
        super(Dense, self).__init__(input_dim, output_dim)

        self.use_bias = use_bias
        self.weight = None
        self.bias = None

        if activation == "sigmoid":
            self.activation = Sigmoid
        elif activation == "relu":
            self.activation = Relu
        elif activation == "softmax":
            self.activation = Softmax
        else:
            raise NotImplementedError

    def init_parameters(self):
        self.weight = xavier_initializer(self.input_dim, self.output_dim)
        if self.use_bias:
            self.bias = zero_initializer(1, self.output_dim)

    def forward(self, inputs):
        # TODO: deal with the cache part
        if self.use_bias:
            affined = np.dot(inputs, self.weight) + self.bias
        else:
            affined = np.dot(inputs, self.weight)

        act = self.activation.forward
        output, cache = act(affined)

        return output, cache

    def backward(self):
        raise NotImplementedError


class Dropout(Module):
    def __init__(self, input_dim, output_dim):
        super(Dropout, self).__init__(input_dim, output_dim)


    def init_parameters(self):
        pass

    def forward(self, inputs):
        pass

    def backward(self):
        pass