"""
The testing here is conducted randomly,
since this is only a warm-up project,
I don't really want to treat it too serious. :)

"""

# test cross entropy loss
from Loss import CrossEntropyLoss
import torch
import numpy as np

X = torch.randn(6, 8, requires_grad=True)
y = torch.tensor([3,4,3,5,1,0])

loss1 = torch.nn.CrossEntropyLoss()
l1 = loss1(X, y)
print(l1)

loss2 = CrossEntropyLoss()
l2 = loss2.forward(X.detach().numpy(), y.numpy())
print(l2)

l1.backward()
print(X.grad)

grads = loss2.backward()
print(grads)

# test sigmoid's forward and backward pass.
from Activations import Sigmoid

X = torch.randn(4,5, requires_grad=True)
sigmoid = torch.nn.Sigmoid()
y = sigmoid(X)
print(y)

y_self, cache_self = Sigmoid.forward(X.detach().numpy())
#print(Sigmoid.forward(X.detach().numpy())[0])

y.sum().backward()
print(X.grad)

grads = Sigmoid.backward(1, cache_self)
print(grads)


# test relu's forward and backward pass.
from Activations import Relu
relu = torch.nn.ReLU()

X = torch.randn(7,8, requires_grad=True)
y = relu(X)
print(y)

y_self, cache_self = Relu.forward(X.detach().numpy())
print(y_self)

y.sum().backward()
print(X.grad)

grads = Relu.backward(np.ones_like(X.detach().numpy()), cache_self)
print(grads)


# test the dense layer
from Layers import Dense

dense = Dense(10, 6, use_bias=True, activation="sigmoid")
x = np.random.randn(4, 10)

dense.init_parameters()
print('---------------Parameters----------------')
print(dense.weight, '\n', dense.bias)
y = dense.forward(x)
print(y)


from torch.nn.functional import linear
weight, bias = torch.tensor(dense.weight, requires_grad=True), torch.tensor(dense.bias, requires_grad=True)
inputs = torch.tensor(x, requires_grad=True)

t_output = linear(inputs, weight.T, bias)
output = sigmoid(t_output)
print(output)

dout = np.ones((4, 6))
grads = dense.backward(dout)

output.sum().backward()
print(grads['dx'])
print(inputs.grad)

print(grads['dW'])
print(weight.grad)

print(grads['db'])
print(bias.grad)





