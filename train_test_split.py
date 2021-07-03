import numpy as np
import pandas as pd
import random


# test size as a percentage of the dataset
# x is a pandas dataframe, y is a numpy array
def train_test_split(x, y, test_size):
    data_size = float(x.shape[0])
    print(f"data contains {data_size} entries")
    test_indexes = random.sample(range(0, int(data_size)), int(test_size*data_size))
    x_test = x.iloc[test_indexes].copy()
    y_test = y[test_indexes]
    x_train = x.drop(x.index[[test_indexes]])
    y_train = np.delete(y, test_indexes)
    return x_train, x_test, y_train, y_test
