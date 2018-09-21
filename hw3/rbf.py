# -*- coding: utf-8 -*-
# @Author: SharonZhu
# @Email: xinyue_zhu[at]berkeley[dot]edu
# @Time: 9/19/18 9:51 AM
# @File: rbf.py

import hw3.utils as utils
import numpy as np
import matplotlib as plt

DATA_DIR = 'starterkit/heart.npz'
LAMBDA = 0.001
SIGMA = [10,3,1,0.3,0.1,0.03]
SPLIT = 0.8

def kernel_rbf(train_x, train_y, val_x, val_y, _lambda, _sigma):
    K = utils.rbf_kernel(train_x, train_x, _sigma)
    K_val = utils.rbf_kernel(val_x, train_x, _sigma)
    C = utils.kernel_ridge_train(K, train_y, _lambda)
    pred_y_train = K.dot(C)
    pred_y_val = K_val.dot(C)

    return utils.mean_square_error(train_y, pred_y_train), utils.mean_square_error(val_y, pred_y_val), C

def main():
    data_x, data_y = utils.load_dataset(DATA_DIR)
    train_x, train_y, val_x, val_y = utils.split_dataset(SPLIT, data_x, data_y)
    s = 0
    for _sigma in SIGMA:
        loss_train, loss_val, C = kernel_rbf(train_x, train_y, val_x, val_y, LAMBDA, _sigma,)
        print('sigma={:10.3f}, train_loss={:10.6f}, val_loss={:10.6f}'.format(_sigma, loss_train, loss_val))
        utils.heatmap(lambda x, y: utils.rbf_kernel(np.column_stack([x, y]), train_x, _sigma) @ C, 5,
                      data_x, data_y,
                      'Results/rbfkernel/heart' + str(s))
        s += 1

if __name__ == '__main__':
    main()