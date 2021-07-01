import numpy as np
from sigmoid import sigmoid, sigmoid_derivative
from loss_functions import sose


class Andromeda:
    def __init__(self, x, y):
        # vector of input values
        self.input = x
        # create a vector of weights for the inputs
        self.weights1 = np.random.rand(self.input.shape[0], 4)
        # weights between hidden and output layers
        self.weights2 = np.random.rand(4, 1)
        # correct outputs
        self.y = y
        # initialise empty output vector
        self.output = np.zeros(y.shape)

    def feed_forward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                                  self.weights2.T) * sigmoid_derivative(self.layer1)))
        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, iterations):
        for i in range(0, iterations):
            self.feed_forward()
            self.backprop()

        print(self.output)