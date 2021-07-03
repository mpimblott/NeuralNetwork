import sys
import warnings
# warnings.filterwarnings("ignore") #suppress warnings
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from train_test_split import train_test_split


headers = ['age', 'sex', 'chest_pain', 'resting_blood_pressure',
        'serum_cholestoral', 'fasting_blood_sugar', 'resting_ecg_results',
        'max_heart_rate_achieved', 'exercise_induced_angina', 'oldpeak', "slope of the peak",
        'num_of_major_vessels', 'thal', 'heart_disease']


def read():
    return pd.read_csv('data/heart.dat', sep=' ', names=headers)


def process():
    heart_df = read()
    # drop the target from the training dataset
    x = heart_df.drop(columns=['heart_disease'])

    # replace data in target with 0/1
    heart_df['heart_disease'] = heart_df['heart_disease'].replace(1, 0)
    heart_df['heart_disease'] = heart_df['heart_disease'].replace(2, 1)

    # convert the targets into a 1d array
    y = heart_df['heart_disease'].values.reshape(x.shape[0], 1)

    # split the data into training and testing data
    x_train, x_test, y_train, y_test = train_test_split(x, y, 0.2)
    print(y_test.shape)


process()
