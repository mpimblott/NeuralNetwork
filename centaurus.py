import sys
import warnings
# warnings.filterwarnings("ignore") #suppress warnings
import numpy as np
import matplotlib.pyplot as plt
import heart
import activation_functions
import loss_functions


x_train, x_test, y_train, y_test = heart.process()


class NeuralNetwork():
    def __init__(self, layers=[13, 8, 1], learning_rate=0.01, iterations=100):
        self.params = {}
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
        self.sample_size = None
        self.layers = layers
        self.x = None
        self.y = None

    def init_weights(self):
        # seed random generator
        np.random.seed(1)
        # seed random weights and biases for each layer
        self.params["w1"] = np.random.rand(self.layers[0], self.layers[1])
        self.params['b1'] = np.random.randn(self.layers[1],)
        self.params['W2'] = np.random.randn(self.layers[1], self.layers[2])
        self.params['b2'] = np.random.randn(self.layers[2],)

    def forward_propagation(self):
        # dot x with weights and add bias
        z1 = self.x.dot(self.params['W1']) + self.params['b1']
        # apply the activation function
        a1 = activation_functions.relu(z1)
        # dot second layer weights with first layer outputs and add bias
        z2 = a1.dot(self.params['W2']) + self.params['b2']
        # apply output function
        yhat = activation_functions.sigmoid(z2)
        # calculate loss
        loss = loss_functions.entropy_loss(self.y, yhat)

        # save calculated parameters
        self.params['z1'] = z1
        self.params['z2'] = z2
        self.params['a1'] = a1

        return yhat, loss
