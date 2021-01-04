import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt


train_data = 'pa2_train.csv'
val_data = 'pa2_valid.csv'
test_data = 'pa2_test_no_label.csv'


def load_data(link):
    my_data = genfromtxt(link, delimiter=',')
    return my_data


def initialize_weights(X):
    theta = np.zeros((1, X.shape[1]))
    return theta


def output_function(Y):
    if Y == 3:
        return 1
    else:
        return -1


def sign_function(data):
    if data <= 0:
        return -1
    else:
        return 1


def preprocess_train_data(data):
    Y = data[:, 0]
    vecfunc = np.vectorize(output_function)
    Y = vecfunc(Y).reshape(len(Y), 1)
    # print(Y)
    X = np.delete(data, [0], axis=1)
    # X = data
    return Y, X


def add_bias(X):
    # print(X.shape[1])
    ones = np.ones((X.shape[0], 1)).astype(float)
    X = np.column_stack([ones, X])
    return X


def plot(iterations, costLists, title, legends, labels):
    # fig, ax = plt.subplots()
    colorsList = ['red', 'blue', 'green']
    print(legends)
    for i in range(0, len(iterations)):
        c = colorsList[i]
        iteration = iterations[i]
        costList = costLists[i]
        plt.plot(costList, iteration, c=c, label=legends[i], markeredgewidth=2)

    plt.ylabel(labels[0])
    plt.xlabel(labels[1])
    plt.legend()
    plt.title(title)
    plt.show()

