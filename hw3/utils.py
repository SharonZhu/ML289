# -*- coding: utf-8 -*-
# @Author: SharonZhu
# @Email: xinyue_zhu[at]berkeley[dot]edu
# @Time: 9/18/18 9:29 PM
# @File: utils.py

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

### For datasets
def load_dataset(data_dir):
    data = np.load(data_dir)
    data_x, data_y = data['x'], data['y']
    data_x /= np.max(data_x)
    return data_x, data_y

def split_dataset(train_p, data_x, data_y):
    train_num = int(data_x.shape[0] * train_p)
    train_x = data_x[:train_num, :]
    train_y = data_y[:train_num]
    val_x = data_x[train_num:, :]
    val_y = data_y[train_num:]
    return train_x, train_y, val_x, val_y


# for features
def gen_features(D, data_x): # generate D degree features
    total_num, feature_num = data_x.shape[0], data_x.shape[1]
    fid = 0
    feature_list = [(np.ones(total_num), 0, 0)]

    while feature_list[fid][1] < D:
        for i in range(int(feature_list[fid][2]), feature_num):
            feature_list.append((feature_list[fid][0] * data_x[:, i], fid+1, i))
        fid += 1
    return np.column_stack(f[0] for f in feature_list)

def gen_feature_special(D, data_x):
    # for 2 features
    xs = []
    for d0 in range(D + 1):
        for d1 in range(D - d0 + 1):
            xs.append((data_x[:, 0] ** d0) * (data_x[:, 1] ** d1))
    return np.column_stack(xs)

# for regression func
def lstsq(A, b):
    return np.linalg.solve(A.T @ A, A.T @ b)

def ridge(A, b, lambda_):
    I = np.eye(A.shape[1])
    return np.linalg.solve(A.T @ A + lambda_ * I, A.T @ b)

def kernel_ridge_train(K, y, _lambda):
    return scipy.linalg.solve(K + _lambda * np.eye(K.shape[0]), y)

# for creating kernels
def poly_kernel(X, Z, degree):
    return (1 + X.dot(Z.T)) ** degree

def rbf_kernel(X, Z, _sigma):
    return np.exp(-sq_dist(X, Z) / (2*_sigma**2))

# for evaluation
def eval(w, feat_train_x, train_y, feat_val_x, val_y):
    # evaluation
    train_error = np.mean((train_y - feat_train_x @ w) ** 2)
    valid_error = np.mean((val_y - feat_val_x @ w) ** 2)
    # prediction for visualization
    pred_y = feat_train_x @ w

    return train_error, valid_error, pred_y

def kernel_eval(train_x, train_y, val_x, val_y, _lambda, p, _sigma, kernel='poly'):
    if kernel == 'poly':
        K = poly_kernel(train_x, train_x, p)
        K_val = poly_kernel(val_x, train_x, p)
    elif kernel == 'rbf':
        K = rbf_kernel(train_x, train_x, _sigma)
        K_val = rbf_kernel(val_x, train_x, _sigma)
    C = kernel_ridge_train(K, train_y, _lambda)
    pred_y_train = K.dot(C)
    pred_y_val = K_val.dot(C)
    return mean_square_error(train_y, pred_y_train), mean_square_error(val_y, pred_y_val)

# helper functions
def mean_square_error(y_act, y_pred):
    return np.mean((y_act-y_pred)**2)

def sq_dist(X, Z):
    return np.add.outer((X**2).sum(1), (Z**2).sum(1)) - 2*X@Z.T

# for plt
def heatmap(f, clip, data_x, data_y, save_name=False):
    # example: heatmap(lambda x, y: x * x + y * y)
    # clip: clip the function range to [-clip, clip] to generate a clean plot
    #   set it to zero to disable this function

    xx0 = xx1 = np.linspace(np.min(data_x), np.max(data_x), 72)
    x0, x1 = np.meshgrid(xx0, xx1)
    x0, x1 = x0.ravel(), x1.ravel()
    z0 = f(x0, x1)

    if clip:
        z0[z0 > clip] = clip
        z0[z0 < -clip] = -clip

    plt.hexbin(x0, x1, C=z0, gridsize=50, cmap=plt.cm.jet, bins=None)
    plt.colorbar()
    cs = plt.contour(
        xx0, xx1, z0.reshape(xx0.size, xx1.size), [-2, -1, -0.5, 0, 0.5, 1, 2], cmap=plt.cm.jet)
    plt.clabel(cs, inline=1, fontsize=10)

    pos = data_y[:] == +1.0
    neg = data_y[:] == -1.0
    plt.scatter(data_x[pos, 0], data_x[pos, 1], c='red', marker='+')
    plt.scatter(data_x[neg, 0], data_x[neg, 1], c='blue', marker='v')
    if save_name:
        plt.savefig(save_name)
    plt.show()

