"""
The testing here is conducted randomly,
since this is only a warm-up project,
I don't really want to treat it too serious. :)

"""

# test cross entropy loss
from Loss import CrossEntropyLoss
import torch

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

# test sigmoid, only the forward pass.
from Activations import Sigmoid

X = torch.randn(4,5)
sigmoid = torch.nn.Sigmoid()
print(sigmoid(X))
print(Sigmoid.forward(X.numpy())[0])



