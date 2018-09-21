# -*- coding: utf-8 -*-
# @Author: SharonZhu
# @Email: xinyue_zhu[at]berkeley[dot]edu
# @Time: 9/18/18 9:11 PM
# @File: d.py

import matplotlib.pyplot as plt
import numpy as np
import hw3.utils as utils
import sys

DATA_DIR = 'starterkit/asymmetric.npz'
LAMBDA = [0.0001, 0.001, 0.01]
P = [5, 6]
SPLIT = 0.8

def sample_train(train_x, train_y, lg_degree):
    indices = np.random.permutation(train_x.shape[0])
    train_id = indices[:int(10**lg_degree)]
    return train_x[train_id,:], train_y[train_id]

if __name__ == '__main__':
    # read dataset
    data_x, data_y = utils.load_dataset(DATA_DIR)
    train_x, train_y, val_x, val_y = utils.split_dataset(SPLIT, data_x, data_y)
    print('training shape:', train_x.shape, train_y.shape)
    val_error_draw = [[{'x':[], 'y':[]} for _ in range(len(LAMBDA))] for _ in range(len(P))]

    for pi, p in enumerate(P):
        feat_train_x = utils.gen_feature_special(p, train_x)
        feat_val_x = utils.gen_feature_special(p, val_x)
        for _lai, _lambda in enumerate(LAMBDA):
            for lg in np.arange(0.5, 5, 0.2):
                # sample multiple times
                val_error = []
                for _ in (range(int(40000/(10**lg)))):
                    sample_train_x, sample_train_y = sample_train(feat_train_x, train_y, lg)
                    print('sample shape:', sample_train_x.shape, sample_train_y.shape)
                    w = utils.ridge(sample_train_x, sample_train_y, _lambda)
                    _, val_error_once, _ = utils.eval(w, feat_train_x, train_y, feat_val_x, val_y)
                    # _, val_error_once = utils.kernel_eval(feat_train_x, train_y, feat_val_x, val_y, _lambda, p)
                    val_error.append(val_error_once)
                val_error_av = np.average(val_error)
                print('p:{:2d}, lambda:{:.5f}, val_error_av:{:10.6f}'.format(p, _lambda, val_error_av))
                val_error_draw[pi][_lai]['x'].append(10**lg)
                val_error_draw[pi][_lai]['y'].append(val_error_av)
    print(val_error_draw)
    print(len(val_error_draw[0][0]['x']), len(val_error_draw[0][0]['y']))
    plt.figure()
    plt.ylim([0, 6])
    plt.xscale('log')
    for i in range(len(P)):
        for j in range(len(LAMBDA)):
            plt.plot(val_error_draw[i][j]['x'], val_error_draw[i][j]['y'], label='p={:2d}, lambda={:.5f}'.format(P[i], LAMBDA[j]))
    plt.legend()
    plt.show()
