# -*- coding: utf-8 -*-
# @Author: SharonZhu
# @Email: xinyue_zhu[at]berkeley[dot]edu
# @Time: 9/17/18 8:58 PM
# @File: kernel_ridge.py
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import sys

data_name = 'asymmetric'
DATA_DIR = 'starterkit/' + data_name + '.npz'

LAMBDA = 0.001
SPLIT = 0.8
P = 16

def mean_square_error(y_act, y_pred):
    return np.mean((y_act-y_pred)**2)

def sq_dist(X, Z):
    return np.add.outer((X**2).sum(1), (Z**2).sum(1)) - 2*X@Z.T

def poly_kernel(X, Z, degree):
    return (1 + X.dot(Z.T)) ** degree

def rbf_kernel(X, Z, _sigma):
    return np.exp(-sq_dist(X, Z) / (2*_sigma**2))

def kernel_ridge_train(K, y, _lambda):
    return scipy.linalg.solve(K + _lambda * np.eye(K.shape[0]), y)

def kernel_eval(train_x, train_y, val_x, val_y, _lambda, p):
    K = poly_kernel(train_x, train_x, p)
    C = kernel_ridge_train(K, train_y, _lambda)
    pred_y_train = K.dot(C)
    K_val = poly_kernel(val_x, train_x, p)
    pred_y_val = K_val.dot(C)
    return mean_square_error(train_y, pred_y_train), mean_square_error(val_y, pred_y_val)

def split_dataset(train_p, data_x, data_y):
    train_num = int(data_x.shape[0] * train_p)
    train_x = data_x[:train_num, :]
    train_y = data_y[:train_num]
    val_x = data_x[train_num:, :]
    val_y = data_y[train_num:]
    return train_x, train_y, val_x, val_y

def main():
    # read dataset
    data = np.load(DATA_DIR)
    data_x, data_y = data['x'], data['y']
    data_x /= np.max(data_x)
    train_x, train_y, val_x, val_y = split_dataset(SPLIT, data_x, data_y)

    # loss
    loss_train = np.empty(P)
    loss_val = np.empty(P)
    print('*********** For {:s} dataset ***********'.format(data_name))
    print('***** 80% + 20% *****')
    for p in range(P):
        loss_train[p], loss_val[p] = kernel_eval(train_x, train_y, val_x, val_y, LAMBDA, p+1)
        print('p={:2d}, train_loss={:10.6f}, val_loss={:10.6f}'.format(p+1, loss_train[p], loss_val[p]))
    # print('training loss:', loss_train)
    # print('validation loss:', loss_val)

if __name__ == '__main__':
    main()