import numpy as np
from Initializers import xavier_initializer, zero_initializer
from Activations import Sigmoid, Relu, Softmax

class Dense:
    def __init__(self, input_dim: int, output_dim: int, use_bias: bool, activation: str) -> None:
        self.input_dim = input_dim
        self.outpupt_dim = output_dim
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
        self.weight = xavier_initializer(self.input_dim, self.outpupt_dim)
        if self.use_bias:
            self.bias = zero_initializer(1, self.outpupt_dim)

    def forward(self, input):
        # TODO: deal with the cache part
        if self.use_bias:
            affined = np.dot(input, self.weight) + self.bias
        else:
            affined = np.dot(input, self.weight)

        act = self.activation.forward
        output, cache = act(affined)

        return output, cache

    def backward(self):
        raise NotImplementedError