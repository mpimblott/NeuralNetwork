import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return np.exp(-x)/((1 + np.exp(-x)) ** 2)


def relu(x):
    return np.maximum(0, x)


def dRelu(x):
    if x > 1:
        return 1
    else:
        return 0
