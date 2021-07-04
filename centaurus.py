import sys
import warnings
# warnings.filterwarnings("ignore") #suppress warnings
import numpy as np
import matplotlib.pyplot as plt
import heart
import activation_functions
import loss_functions


# following https://heartbeat.fritz.ai/building-a-neural-network-from-scratch-using-python-part-1-6d399df8d432
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
        self.params['w2'] = np.random.randn(self.layers[1], self.layers[2])
        self.params['b2'] = np.random.randn(self.layers[2],)

    def forward_propagation(self):
        # dot x with weights and add bias
        z1 = self.x.dot(self.params['w1']) + self.params['b1']
        # apply the activation function
        a1 = activation_functions.relu(z1)
        # dot second layer weights with first layer outputs and add bias
        z2 = a1.dot(self.params['w2']) + self.params['b2']
        # apply output function
        yhat = activation_functions.sigmoid(z2)
        # calculate loss
        loss = loss_functions.entropy_loss(self.y, yhat)

        # save calculated parameters
        self.params['z1'] = z1
        self.params['z2'] = z2
        self.params['a1'] = a1

        return yhat, loss

    def back_propagation(self, yhat):
        '''
        Computes the derivatives and update weights and bias according.
        '''
        y_inv = 1 - self.y
        yhat_inv = 1 - yhat

        dl_wrt_yhat = np.divide(y_inv, loss_functions.eta(yhat_inv)) - np.divide(self.y, loss_functions.eta(yhat))
        dl_wrt_sig = yhat * yhat_inv
        dl_wrt_z2 = dl_wrt_yhat * dl_wrt_sig

        dl_wrt_a1 = dl_wrt_z2.dot(self.params['w2'].T)
        dl_wrt_w2 = self.params['a1'].T.dot(dl_wrt_z2)
        dl_wrt_b2 = np.sum(dl_wrt_z2, axis=0, keepdims=True)

        dl_wrt_z1 = dl_wrt_a1 * activation_functions.dRelu(self.params['z1'])
        dl_wrt_w1 = self.x.T.dot(dl_wrt_z1)
        dl_wrt_b1 = np.sum(dl_wrt_z1, axis=0, keepdims=True)

        # update weights and biases
        self.params['w1'] = self.params['w1'] - self.learning_rate * dl_wrt_w1
        self.params['w2'] = self.params['w2'] - self.learning_rate * dl_wrt_w2
        self.params['b1'] = self.params['b1'] - self.learning_rate * dl_wrt_b1
        self.params['b2'] = self.params['b2'] - self.learning_rate * dl_wrt_b2

    def fit(self, x, y):
        self.x = x
        self.y = y
        self.init_weights()

        for i in range(self.iterations):
            yhat, loss = self.forward_propagation()
            self.back_propagation(yhat)
            self.loss.append(loss)

    def predict(self, x):
        z1 = x.dot(self.params['w1']) + self.params['b1']
        a1 = activation_functions.relu(z1)
        z2 = a1.dot(self.params['w2']) + self.params['b2']
        pred = activation_functions.sigmoid(z2)
        return(pred)


x_train, x_test, y_train, y_test = heart.process()
nn = NeuralNetwork()
nn.fit(x_train, y_train)
