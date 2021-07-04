import numpy as np


# sum of squares error ---------------------
# y vector of correct outputs
# yhat output layer
def sose(y, yhat):
    n = y.shape[0]
    sigma = 0
    for i in range(0, n):
        sigma += (y[i] - yhat[i])**2

    return sigma


# cross-entropy loss -----------------------
def eta(x):
    ETA = 0.0000000000001
    return np.maximum(x, ETA)


def entropy_loss(y, yhat):
    nsample = len(y)
    yhat_inv = 1.0 - yhat
    y_inv = 1.0 - y
    yhat = eta(yhat)
    yhat_inv = eta(yhat_inv)
    loss = -1 / nsample * (np.sum(np.multiply(np.log(yhat), y) + np.multiply(y_inv, np.log(yhat_inv))))
    return loss
