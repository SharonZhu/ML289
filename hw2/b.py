#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
import sys

# There is numpy.linalg.lstsq, which you should use outside of this classs
def lstsq(A, b):
    return np.linalg.solve(A.T @ A, A.T @ b)

def error(X, y, n):
    w = lstsq(X, y)
    y_pred = X @ w
    e = (np.linalg.norm(y-y_pred) ** 2) / n
    return e

def main():
    data = spio.loadmat('data/1D_poly.mat', squeeze_me=True)
    x_train = np.array(data['x_train'])
    y_train = np.array(data['y_train']).T
    # print(x_train, x_train.shape)
    # print(y_train, y_train.shape)
    # sys.exit()

    n = 20  # max degree
    err = np.zeros(n-1)
    # print(err.shape)

    # fill in err
    # YOUR CODE HERE
    # build [1,1,...1], [x1, x2, ..., xn], [x1^2, ..., xn^2] ... [x1^D-1, ..., xn^D-1]
    X = np.ones(shape=[n, 1])  # first cols with 1
    for d in range(1, n):
        print('d',d)
        Xi = np.array([x_train[i]**d for i in range(n)]).T
        # print(Xi.shape)
        X = np.column_stack((X, Xi))
        # print(X.shape)

        # cal error for current X with D
        err[d-1] = error(X, y_train, n)
        # print(err[d-1])

    plt.plot(err)
    plt.xlabel('Degree of Polynomial')
    plt.ylabel('Training Error')
    plt.show()


if __name__ == "__main__":
    main()
