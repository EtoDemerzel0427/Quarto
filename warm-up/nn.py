from Layers import Dense
from typing import List

class DNN:
    def __init__(self, layer_dims: List) -> None:
        self.layer_dims = layer_dims
        self.dim = len(self.layer_dims)

    def init_parameters(self):
        raise NotImplementedError

    def forward(self, input):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


