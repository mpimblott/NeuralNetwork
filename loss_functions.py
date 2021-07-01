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
