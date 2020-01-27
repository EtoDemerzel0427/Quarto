import numpy as np
from Layers import Dense
from typing import List


class DNNClassfier:
    def __init__(self) -> None:
        self.layers = []
        self.grads = []

    def init_parameters(self):
        for layer in self.layers:
            layer.init_parameters()

    def add(self, layer):
        self.layers.append(layer)

    def zero_grad(self):
        self.grads = []

    def forward(self, inputs):
        x = inputs.copy()
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def backward(self, dout):
        self.zero_grad()

        grads = dout.copy()
        for layer in reversed(self.layers):
            layer_grads = layer.backward(grads)
            if layer.use_bias:
                dW, db = layer_grads['dW'], layer_grads['db']
                self.grads.append({'dW': dW, 'db': db})
            else:
                dW = layer_grads['dW']
                self.grads.append({'dW': dW})


            grads = layer_grads['dx']

        self.grads.reverse()

    def train_step(self, lr=1e-3, weight_decay=0):
        """
        a simple SGD optimizer.
        :param lr: learning rate.
        :param weight_decay: weight decay.
        :return:
        """
        for i, layer in enumerate(self.layers):
            if layer.use_bias:
                db = self.grads[i]['db']
                db += weight_decay * db
                layer.bias -= lr * db

            dW = self.grads[i]['dW']
            dW += weight_decay * dW
            layer.weight -= lr * dW

    def eval(self, inputs):
        x = inputs.copy()
        for layer in self.layers:
            x = layer.eval(x)

        output = np.argmax(x, axis=1)
        return output


if __name__ == '__main__':
    x = np.random.randn(4, 50)
    net = DNNClassfier()
    net.add(Dense(50, 20, use_bias=True))
    net.add(Dense(20, 15, use_bias=True))
    net.add(Dense(15, 10, use_bias=True, activation="Linear"))

    net.init_parameters()
    output = net.forward(x)
    from Loss import CrossEntropyLoss
    criterion = CrossEntropyLoss()
    target = np.array([1,3,4,6])
    loss = criterion.forward(output, target)
    print(loss)
    grads = criterion.backward()
    print(grads)
    net.backward(grads)
    net.train_step()


