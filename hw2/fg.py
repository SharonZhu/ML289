import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
import sys

data = spio.loadmat('data/polynomial_regression_samples.mat', squeeze_me=True)
data_x = data['x']
data_y = data['y']
Kc = 4  # 4-fold cross validation
KD = 6  # max D = 6
LAMBDA = [0, 0.05, 0.1, 0.15, 0.2]

valid_num = int(data_x.shape[0] / Kc)
train_num = valid_num * (Kc - 1)
feature_num = data_x.shape[1]
total_num = data_x.shape[0]


def ridge(A, b, lambda_):
    I = np.eye(A.shape[1])
    return np.linalg.solve(A.T @ A + lambda_ * I, A.T @ b)

def genFeatures(D): # generate D degree features
    fid = 0
    feature_list = [(np.ones(total_num), 0, 0)]

    while feature_list[fid][1] < D:
        for i in range(int(feature_list[fid][2]), feature_num):
            feature_list.append((feature_list[fid][0] * data_x[:, i], fid+1, i))
        fid += 1
    return np.column_stack(f[0] for f in feature_list)

def fit(D, lambda_):
    # YOUR CODE TO COMPUTE THE AVERAGE ERROR PER SAMPLE
    # Error for train and validation
    train_error = np.zeros(4)
    valid_error = np.zeros(4)

    feature_x = genFeatures(D+1)
    # print(feature_x.shape)

    for kc in range(Kc):
        valid_x = feature_x[kc*valid_num:(kc+1)*valid_num]
        valid_y = data_y[kc*valid_num:(kc+1)*valid_num]
        train_x = np.delete(feature_x, list(range(kc*valid_num, (kc+1)*valid_num)), axis=0)
        train_y = np.delete(data_y, list(range(kc*valid_num, (kc+1)*valid_num)))

        # train
        w_ridge = ridge(train_x, train_y, lambda_)

        # evaluation
        train_error[kc] = np.mean((train_y - train_x @ w_ridge) ** 2)
        valid_error[kc] = np.mean((valid_y - valid_x @ w_ridge) ** 2)

    return np.mean(train_error), np.mean(valid_error)

def main():
    np.set_printoptions(precision=11)
    Etrain = np.zeros((KD, len(LAMBDA)))
    Evalid = np.zeros((KD, len(LAMBDA)))
    # Etrain = np.zeros(KD)
    # Evalid = np.zeros(KD)

    # np.random.shuffle(data_x)  # shuffle data for splitting

    for D in range(KD):
        print(D)
        for i in range(len(LAMBDA)):
            Etrain[D, i], Evalid[D, i] = fit(D, LAMBDA[i])
        # Etrain[D], Evalid[D] = fit(D, 0.1)
    print('Average train error:', Etrain, sep='\n')
    print('Average valid error:', Evalid, sep='\n')

    # YOUR CODE to find best D and i
    # print(Evalid.argmin())
    D, i = divmod(Evalid.argmin(), len(LAMBDA))
    print(D, i)

if __name__ == "__main__":
    main()
