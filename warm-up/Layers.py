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

    def backward(self, dout):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


class Dense(Module):
    def __init__(self, input_dim: int, output_dim: int, use_bias: bool = False, activation: str = "relu") -> None:
        super(Dense, self).__init__(input_dim, output_dim)

        self.use_bias = use_bias
        self.weight = None
        self.bias = None
        self.cache = {}

        if activation == "Linear":
            self.activation = None
        elif activation == "sigmoid":
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
        """

        :param inputs: fed data, m x input_dim, m is the batch size.
        :return:
        """
        # TODO: deal with the cache part
        if self.use_bias:
            affined = np.dot(inputs, self.weight) + self.bias
            self.cache['W'] = self.weight
            self.cache['b'] = self.bias
            self.cache['x'] = inputs
        else:
            affined = np.dot(inputs, self.weight)
            self.cache['W'] = self.weight
            self.cache['x'] = inputs

        if self.activation is not None:
            act = self.activation.forward
            output, act_cache = act(affined)
            self.cache['act'] = act_cache
        else:
            output = affined

        return output

    def backward(self, dout):
        """
        :param dout: The gradient of the output. shape: m x output_dim, m is the batch size.
        :return:
        """

        # backprop of activation
        if self.activation is not None:
            act_back = self.activation.backward
            d_linear = act_back(dout, self.cache['act'])
        else:
            d_linear = dout

        # backprop of linear layer
        dW = self.cache['x'].T @ d_linear
        dx = d_linear @ (self.cache['W'].T)
        if self.bias is not None:
            db = np.sum(d_linear, axis=0, keepdims=True)

            grads = {'dW': dW, 'dx': dx, 'db': db}
        else:
            grads = {'dW': dW, 'dx': dx}

        return grads

    def eval(self, inputs):
        """
        Simply do what the forward method do, but do not touch the cache.
        :param inputs:
        :return:
        """
        if self.use_bias:
            affined = np.dot(inputs, self.weight) + self.bias
        else:
            affined = np.dot(inputs, self.weight)

        if self.activation is not None:
            act = self.activation.forward
            output, act_cache = act(affined)
        else:
            output = affined

        return output

    def __repr__(self):
        return f"Dense(input_dim={self.input_dim}, output_dim={self.output_dim}, use_bias={self.use_bias}," \
            f"activation={self.activation})"



class Dropout(Module):
    def __init__(self, input_dim, output_dim):
        super(Dropout, self).__init__(input_dim, output_dim)


    def init_parameters(self):
        pass

    def forward(self, inputs):
        pass

    def backward(self, dout):
        pass

    def __repr__(self):
        raise NotImplementedError


if __name__ == '__main__':
    dense = Dense(10, 6, use_bias=True, activation="sigmoid")
    x = np.random.randn(4, 10)

    dense.init_parameters()
    print('---------------Parameters----------------')
    print(dense.weight, '\n', dense.bias)
    y = dense.forward(x)
    print(y)

    dout = np.ones((4, 6))
    print(dense.backward(dout))
