#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio

# There is numpy.linalg.lstsq, whicn you should use outside of this classs
def lstsq(A, b):
    return np.linalg.solve(A.T @ A, A.T @ b)


def main():
    data = spio.loadmat('data/1D_poly.mat', squeeze_me=True)
    x_train = np.array(data['x_train'])
    y_train = np.array(data['y_train']).T
    y_fresh = np.array(data['y_fresh']).T

    n = 20  # max degree
    err_train = np.zeros(n - 1)
    err_fresh = np.zeros(n - 1)

    # fill in err_fresh and err_train
    # YOUR CODE HERE
    X = np.ones(shape=[n, 1])  # first cols with 1
    for d in range(1, n):
        print('d',d)
        Xi = np.array([x_train[i]**d for i in range(n)]).T
        # print(Xi.shape)
        X = np.column_stack((X, Xi))
        # print(X.shape)

        w = lstsq(X, y_train)
        y_pred = X @ w
        err_train[d-1] = (np.linalg.norm(y_train-y_pred) ** 2) / n
        err_fresh[d-1] =  (np.linalg.norm(y_fresh-y_pred) ** 2) / n

    plt.figure()
    plt.ylim([0, 6])
    plt.plot(err_train, label='train')
    plt.plot(err_fresh, label='fresh')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
