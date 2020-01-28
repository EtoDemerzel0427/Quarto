import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nn import DNNClassfier
from Layers import Dense
from Loss import CrossEntropyLoss

def accuracy(inputs, targets):
    if not isinstance(inputs, np.ndarray):
        inputs = np.array(inputs)
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)

    assert len(inputs.shape) == 1
    assert len(targets.shape) == 1
    return np.mean(inputs==targets)

# Load the data
# data can be downloaded here: https://pjreddie.com/projects/mnist-in-csv/
train = pd.read_csv("./mnist_train.csv", header=None)
test = pd.read_csv("./mnist_test.csv", header=None)

Y_train = train[0]
X_train = train.drop(labels=0, axis="columns")

del train

# Depreciated: labels to one-hot vectors
# classes = np.eye(10)
# Y_train = classes[Y_train, :]

# normalize X_train
X_train = X_train.values / 255.

random_seed = None # todo: set a random seed
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)

print(X_train.shape, Y_train.shape)

# define the model
net = DNNClassfier()
net.add(Dense(X_train.shape[1], 256, use_bias=False, activation="relu"))
net.add(Dense(256, 128, use_bias=False, activation="relu"))
net.add(Dense(128, 64, use_bias=False, activation="relu"))
net.add(Dense(64, 32, use_bias=False, activation="relu"))
net.add(Dense(32, 16, use_bias=False, activation="relu"))
net.add(Dense(16, 10, use_bias=False, activation="Linear"))
net.init_parameters()

criterion = CrossEntropyLoss()

# training
batch_size = 1000
epoch = 300

batch_num = int(np.ceil(X_train.shape[0]/batch_size))

for _ in range(epoch):
    for i in range(batch_num):
        data = X_train[i*batch_size: (i+1) * batch_size]
        target = Y_train[i*batch_size: (i+1) * batch_size]

        output = net.forward(data)
        loss = criterion.forward(output, target)
        #print(loss)

        grads = criterion.backward()
        net.backward(grads)

        net.train_step(lr=1e-2)
    print(f"epoch {_ + 1} train acc: {accuracy(net.eval(X_train), Y_train)}    val acc: {accuracy(net.eval(X_val), Y_val)}")

print(f"Val Acc: {accuracy(net.eval(X_val), Y_val)}")
net.save_model("weights.npy")


# test
Y_test = test[0]
X_test = test.drop(labels=0, axis="columns")

del test

# normalize X_train
X_test = X_test.values / 255.
print(f"Test Acc: {accuracy(net.eval(X_test), Y_test)}")



