import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
import sys

data_name = 'asymmetric'
DATA_DIR = 'starterkit/' + data_name + '.npz'
data = np.load(DATA_DIR)
data_x = data['x']
data_x /= np.max(data_x)
data_y = data['y']

Kc = 5  # 5-fold cross validation
KD = 16  # max D = 16
LAMBDA = 0.001
SPLIT = 0.8

total_num = data_x.shape[0]
feature_num = data_x.shape[1]

def split_dataset(train_p, data_x, data_y):
    train_num = int(data_x.shape[0] * train_p)
    train_x = data_x[:train_num, :]
    train_y = data_y[:train_num]
    val_x = data_x[train_num:, :]
    val_y = data_y[train_num:]
    return train_x, train_y, val_x, val_y

def ridge(A, b, lambda_):
    I = np.eye(A.shape[1])
    return np.linalg.solve(A.T @ A + lambda_ * I, A.T @ b)

def genFeatures(x, D): # generate D degree features from x1 and x2
    xs = []
    for d0 in range(D + 1):
        for d1 in range(D - d0 + 1):
            xs.append((x[:, 0] ** d0) * (x[:, 1] ** d1))
    return np.column_stack(xs)

def fit(D, lambda_):
    # YOUR CODE TO COMPUTE THE AVERAGE ERROR PER SAMPLE
    # Error for train and validation
    train_x, train_y, val_x, val_y = split_dataset(SPLIT, data_x, data_y)

    feat_x_train = genFeatures(train_x, D)
    feat_x_val = genFeatures(val_x, D)
    # print('feature shape', feat_x_train.shape)
    # print('feature shape', feat_x_val.shape)

    # train
    w_ridge = ridge(feat_x_train, train_y, lambda_)

    # evaluation
    train_error = np.mean((train_y - feat_x_train @ w_ridge) ** 2)
    valid_error = np.mean((val_y - feat_x_val @ w_ridge) ** 2)

    # prediction for visualization
    pred_y = feat_x_train @ w_ridge

    return w_ridge, train_error, valid_error, pred_y

def heatmap(f, save_dir, clip=5):
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
    plt.savefig(save_dir)
    plt.show()

def main():
    np.set_printoptions(precision=11)
    # Etrain = np.zeros(KD)
    # Evalid = np.zeros(KD)
    # np.random.shuffle(data_x)  # shuffle data for splitting
    print('*********** For {:s} dataset ***********'.format(data_name))
    for D in range(1,17):
        # print('current polynomial order: ', D)
        w, Etrain, Evalid, pred = fit(D, LAMBDA)
        if D in [2,4,6,8,10,12]:
            heatmap(lambda x,y: genFeatures(np.vstack([x, y]).T, D) @ w, save_dir='Results/polyridge/' + data_name + '_' + str(D))
        print('p={:2d}, train_loss={:10.6f}, val_loss={:10.6f}'.format(D, Etrain, Evalid))
    # print('Average train error:', Etrain, sep='\n')
    # print('Average valid error:', Evalid, sep='\n')


if __name__ == "__main__":
    main()
