import numpy as np
import matplotlib.pyplot as plt
import sys

# Load the training dataset
train_features = np.load("data/train_features.npy")
train_labels = np.load("data/train_labels.npy").astype("int8")

n_train = train_labels.shape[0]
p = 5000

def visualize_digit(features, label):
    # Digits are stored as a vector of 400 pixel values. Here we
    # reshape it to a 20x20 image so we can display it.
    plt.imshow(features.reshape(20, 20), cmap="binary")
    plt.xlabel("Digit with label " + str(label))


# Visualize a digit
# visualize_digit(train_features[0,:], train_labels[0])

# TODO: Plot three images with label 0 and three images with label 1
pos = train_labels[:] == 1
neg = train_labels[:] == 0

plt.figure()
for i in range(3):
    plt.subplot(1, 3, i+1)
    visualize_digit(train_features[pos,:][i], 1)

plt.figure()
for i in range(3):
    plt.subplot(1, 3, i+1)
    visualize_digit(train_features[neg,:][i], 0)

# plt.show()
# Linear regression

# TODO: Solve the linear regression problem, regressing
# X = train_features against y = 2 * train_labels - 1
X = train_features
y = 2 * train_labels - 1
def linear_regression(X, y):
    w = np.linalg.solve(X.T @ X, X.T @ y)
    ans = np.linalg.norm(X@w - y) ** 2
    return w, ans


# TODO: Report the residual error and the weight vector
w, ans = linear_regression(X, y)
print('w: ', w)
print('||Xw-y||_2^2: ', ans)


# Load the test dataset
# It is good practice to do this after the training has been
# completed to make sure that no training happens on the test
# set!
test_features = np.load("data/test_features.npy")
test_labels = np.load("data/test_labels.npy").astype("int8")

n_test = test_labels.shape[0]

# TODO: Implement the classification rule and evaluate it
# on the training and test set
def natural_rule(X, y, weight):
    pred_label = X@weight
    real_pos = y[:] == 1
    pred_pos = pred_label[:] > 0
    # p_neg = pred_label[:] <= 0
    error = np.sum(real_pos ^ pred_pos)
    correct = y.shape[0] - error
    right_perc = correct / y.shape[0]
    error_perc = error / y.shape[0]
    # print(perct)
    return right_perc, error_perc

c, r = natural_rule(train_features, train_labels, w)
print('training set: correct = {:.20f}, error = {:.20f}'.format(c, r))
ct, rt = natural_rule(test_features, test_labels, w)
print('testing set: correct = {:.20f}, error = {:.20f}'.format(ct, rt))

# TODO: Implement classification using random features
def gen_random_feature(X):
    n, d = X.shape[0], X.shape[1]
    G = np.random.normal(0, 0.01, [d, p])
    b = np.random.uniform(0, 2 * np.pi, p)
    print(np.cos(X@G + b).shape)
    return np.cos(X@G + b)

random_train_features = gen_random_feature(train_features)
random_test_features = gen_random_feature(test_features)

random_w, random_ans = linear_regression(random_train_features, y)
print('w: ', random_w)
print('ans: ', random_ans)

random_c, random_r = natural_rule(random_train_features, train_labels, random_w)
print('training set: correct = {:.20f}, error = {:.20f}'.format(random_c, random_r))
random_ct, random_rt = natural_rule(random_test_features, test_labels, random_w)
print('testing set: correct = {:.20f}, error = {:.20f}'.format(random_ct, random_rt))
