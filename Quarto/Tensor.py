import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad

    def backward(self, grad):
        raise NotImplementedError

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"



