import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from ml_iml2_util import sign_function, output_function, initialize_weights, load_data, preprocess_train_data, add_bias, plot, train_data, test_data, val_data
import datetime

powers = [1, 2, 3, 7, 15]


def generate_k_matrix(X, Y, p):
    # K = np.zeros((X.shape[0], X.shape[0]))
    K = np.matmul(X, Y.transpose())
    K = 1 + K
    # print(K[0, 0], " ", K[0, 1])
    K = np.power(K, p)
    # print(K[0, 0], " ", K[0, 1])
    return K


def get_kernelized_perceptron_model(iterations, p):
    Y, X = preprocess_train_data(load_data(train_data))
    X = add_bias(X)
    K = generate_k_matrix(X, X, p)
    # kernel_function(X[0,:], X[1,:], 2)
    n = len(Y)
    print(K.shape)
    a = np.zeros((n, 1))
    iter = 0
    alphaList = []
    accuracyList = []
    iterationList = []
    while iter < iterations:
        print(iter)
        t = 0
        for i in range(0, n):
            # print(1)
            # an = datetime.datetime.now()
            product = K[i, :].transpose() * a * Y
            # b = datetime.datetime.now()
            # print(2, " : ", (b - an).microseconds)
            prediction_raw = np.sum(product)
            # c = datetime.datetime.now()
            # print(3, " : ", (c - b).microseconds)
            u = np.sign(prediction_raw)
            # d = datetime.datetime.now()
            # print(4, " : ", (d - c).microseconds)
            if Y[i] * u <= 0:
                a[i] = a[i] + 1
                t = t + 1
        accuracyList.append((1 - t/n) * 100)
        iterationList.append(iter)
        alphaList.append(a)

    print(alphaList)
    return alphaList, X, Y, accuracyList, iterationList


def validate_data(alphaList, X_train, Y_train, p):
    Y, X = preprocess_train_data(load_data(val_data))
    X = add_bias(X)
    K = generate_k_matrix(X, X_train, p)
    n = len(Y)
    accuracyList = []
    iterationList = []
    for a in alphalist:
        for i in range(0, n):
            product = K[i, :].transpose() * a * Y
            # b = datetime.datetime.now()
            # print(2, " : ", (b - an).microseconds)
            prediction_raw = np.sum(product)
            # c = datetime.datetime.now()
            # print(3, " : ", (c - b).microseconds)
            u = np.sign(prediction_raw)
            if Y[i] * u <= 0:
                t = t + 1

        accuracyList.append((1 - t / n) * 100)
        iterationList.append(i)

    return accuracyList, iterationList

for p in powers:
    alphalist, X_train, Y_train, accuracyList_train, iterationList_train = get_kernelized_perceptron_model(10, p)
    accuracyList_val, iterationList_val = validate_data(alphalist, X_train, Y_train, p)
    accuracy = [accuracyList_train, accuracyList_val]
    iterations = [iterationList_train, iterationList_val]
    legends = ["Training Data", "Validation Data"]
    labels = ["Accuracy", "Iterations'"]
    plot(iterations, accuracy, "Accuracy vs Iterations", legends, labels)

# plot([powers], accuracy)
