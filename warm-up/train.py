import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
