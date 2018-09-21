# -*- coding: utf-8 -*-
# @Author: SharonZhu
# @Email: xinyue_zhu[at]berkeley[dot]edu
# @Time: 9/16/18 6:56 PM
# @File: visualize_data.py

import matplotlib.pyplot as plt
import numpy as np
import sys

def visualize(data_dir):
    data = np.load(data_dir)
    data_x, data_y = data['x'], data['y']
    print(data_x.shape, data_y.shape)
    data_name = data_dir.split('/')[1].split('.')[0]
    print(data_name)

    pos = data_y[:] == +1.0
    neg = data_y[:] == -1.0
    plt.scatter(data_x[pos, 0], data_x[pos, 1], c='red', marker='+')
    plt.scatter(data_x[neg, 0], data_x[neg, 1], c='blue', marker='v')
    plt.savefig('Results/' + data_name + '.png')
    plt.show()

if __name__ == '__main__':
    visualize('starterkit/circle.npz')
    visualize('starterkit/heart.npz')
    visualize('starterkit/asymmetric.npz')