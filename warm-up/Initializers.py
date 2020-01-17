"""
Reference:
https://towardsdatascience.com/
weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79
"""

import numpy as np

def xavier_initializer(fan_in, fan_out):
    return np.random.uniform(-1, 1, size=(fan_in, fan_out)) * np.sqrt(6. / (fan_in + fan_out))


def kaiming_initializer(fan_in, fan_out):
    return np.random.randn(fan_in, fan_out) * np.sqrt(2. / fan_in)


def zero_initializer(fan_in, fan_out):
    return np.zeros((fan_in, fan_out))


