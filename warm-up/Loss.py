"""
In this simple version, I assume loss to be the same as an activation function,
from which we only need their values and grads.
"""

import numpy as np
from typing import Tuple


class Loss:
    def __init__(self):
        self.cache = None

    def forward(self, inputs: np.ndarray, target: np.ndarray) -> Tuple[float, Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError

# TODO: add type and shape checking
class CrossEntropyLoss(Loss):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, inputs: np.ndarray, target: np.ndarray) -> float:
        """

        :param input: N x D ndarray, each row as a vector.
        If a vector with (D,) shape is provided, we convert
        it to a (1,D) array.
        :param target:  (N,) ndarray, each element represent
        a class index, i.e., integer ranging from 0 to D-1 (included).
        :return: a scalar, the loss value.
        """
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)
        # if len(target.shape) == 1:
        #     target = target.reshape(1, -1)

        # class_idx = np.argmax(target, axis=1)

        # scale the input
        input_max = np.max(inputs, axis=1, keepdims=True)  # N x 1
        inputs -= input_max

        input_exp = np.exp(inputs)
        loss = -inputs[np.arange(len(inputs)), target][:, None] + np.log(np.sum(input_exp, axis=1, keepdims=True))
        loss = np.mean(loss).squeeze()

        self.cache = input_exp, target

        return loss

    def backward(self, dout=1):
        input_exp, target = self.cache
        grads = input_exp / np.sum(input_exp, axis=1, keepdims=True)

        grads[np.arange(len(grads)), target] -= 1

        return dout * grads / len(grads)


if __name__ == '__main__':
    loss = CrossEntropyLoss()
    # x = np.random.randn(4, 10)
    # y = np.eye(10)[[1,5, 6, 7], :]
    x = np.array([[-0.3680,  1.4395, -0.8441, -1.2680, -0.6068],
        [-1.3705, -1.4921, -0.0172, -0.5453, -0.8409],
        [-0.2652, -0.3018, -0.2923, -0.5061,  1.3517]])
    y = np.array([3,4,3])

    l = loss.forward(x, y)
    grads = loss.backward()
    print(l)
    print(grads)

# [[ 0.0374,  0.2280,  0.0232, -0.3181,  0.0295],
#  [ 0.0342,  0.0303,  0.1325,  0.0781, -0.2752],
#  [ 0.0381,  0.0367,  0.0370, -0.3034,  0.1917]]

