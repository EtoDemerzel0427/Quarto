import numpy as np
import pandas as pd
from nn import DNNClassfier
from Layers import Dense

def accuracy(inputs, targets):
    if not isinstance(inputs, np.ndarray):
        inputs = np.array(inputs)
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)

    assert len(inputs.shape) == 1
    assert len(targets.shape) == 1
    return np.mean(inputs==targets)

test = pd.read_csv("./mnist_test.csv", header=None)

# test
Y_test = test[0]
X_test = test.drop(labels=0, axis="columns")

del test

# normalize X_train
X_test = X_test.values / 255.


net = DNNClassfier()
net.add(Dense(X_test.shape[1], 256, use_bias=False, activation="relu"))
net.add(Dense(256, 128, use_bias=False, activation="relu"))
net.add(Dense(128, 64, use_bias=False, activation="relu"))
net.add(Dense(64, 32, use_bias=False, activation="relu"))
net.add(Dense(32, 16, use_bias=False, activation="relu"))
net.add(Dense(16, 10, use_bias=False, activation="Linear"))

weights = np.load("weights.npy", allow_pickle=True)

net.assign_parameters(weights)
print(f"Test Acc: {accuracy(net.eval(X_test), Y_test)}")