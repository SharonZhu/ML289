import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import sklearn.linear_model
from sklearn.model_selection import train_test_split
import sys


######## EMBEDDING FUNCTIONS ##########
######## YOUR CODE FOR (c) ##########


## Input: original dimension d,
## Input: embedding dimension k
## Output: d x k random
## Gaussian matrix J with entry-wise
## variances 1/k so that, 
## for any row vector z^T in R^d, 
## z^T J  is a random features embedding for z^T
def random_JL_matrix(d, k):
    # your code here
    randmatrix = np.random.normal(0., 1., (d, k))
    return 1./np.sqrt(k) * randmatrix


## Input: n x d data matrix X
## Input: embedding dimension k
## Output: d x k matrix V
## with orthonormal columns 
## corresponding to the top k right signular vectors 
## of X. Thus, for a row vector z^T in R^d
## z^T V  is the projection of z^T 
## onto the the top k right-singular vectors of X,
def pca_embedding_matrix(X, k):
    # your code here
    U, S, V = np.linalg.svd(X)
    return (V.T)[:, :k]


######## END YOUR CODE FOR (c) ##########


# applies the linear transformation N 
# to the rows of X, via X.dot(N)
def linear_feature_transform(X, N):
    return X.dot(N)


# uses pca_embedding_matrix method
# to embed the rows of X onto the first
# k principle components
def pca_embedding(X, k):
    P = pca_embedding_matrix(X, k)
    return linear_feature_transform(X, P)


# uses random_JL_matrix method
# to transform rows of X
# by a k-dimensional JL transform
def random_embeddings(X, k):
    P = random_JL_matrix(X.shape[1], k)
    return linear_feature_transform(X, P)


######### LINEAR MODEL FITTING ############
######### DO NOT ALTER ##################
def rand_embed_accuracy_split(X, y, k):
    '''
    Fitting a k dimensional feature set obtained
    from random embedding of X, versus y
    for binary classification for y in {-1, 1}
    '''

    # test train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # random embedding
    _, d = X.shape
    J = random_JL_matrix(d, k)
    rand_embed_X = linear_feature_transform(X_train, J)

    # fit a linear model
    line = sklearn.linear_model.LinearRegression(fit_intercept=False)
    line.fit(rand_embed_X, y_train)

    # predict y
    y_pred = line.predict(linear_feature_transform(X_test, J))

    # return the test error
    return 1 - np.mean(np.sign(y_pred) != y_test)


def pca_embed_accuracy_split(X, y, k):
    '''
    Fitting a k dimensional feature set obtained
    from PCA embedding of X, versus y
    for binary classification for y in {-1, 1}
    '''

    # test-train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # pca embedding
    V = pca_embedding_matrix(X_train, k)
    pca_embed_X = linear_feature_transform(X_train, V)

    # fit a linear model
    line = sklearn.linear_model.LinearRegression(fit_intercept=False)
    line.fit(pca_embed_X, y_train)

    # predict y
    y_pred = line.predict(linear_feature_transform(X_test, V))

    # return the test error
    return 1 - np.mean(np.sign(y_pred) != y_test)


######## LOADING THE DATASETS #########

# to load the data:
data = np.load('data/data3.npz')
X = data['X']
y = data['y']
n, d = X.shape

n_trials = 10  # to average for accuracies over random embeddings

######### YOUR CODE GOES HERE ##########

# Using PCA and Random embedding for:
# Visualizing the datasets
pca_vis = pca_embedding(X, 2)
rand_vis = random_embeddings(X, 2)

# Computing the accuracies over different datasets.
pca_acc = np.zeros(d)
rand_acc = np.zeros((n_trials, d))

for k in range(1, d+1):
    pca_acc[k-1] = pca_embed_accuracy_split(X, y, k)
    for t in range(n_trials):
        rand_acc[t, k-1] = rand_embed_accuracy_split(X, y, k)

rand_acc = np.mean(rand_acc, axis=0)


# Don't forget to average the accuracy for multiple
# random embeddings to get a smooth curve.





# And computing the SVD of the feature matrix

U, S, V = np.linalg.svd(X)

######## YOU CAN PLOT THE RESULTS HERE ########

# plt.plot, plt.scatter would be useful for plotting
# visualizing top-2 pca
# plt.figure()
# plt.scatter(pca_vis[:, 0], pca_vis[:, 1], c=y)
# plt.title('Top-k PCA')
#
# # visualizing random projection
# plt.figure()
# plt.scatter(rand_vis[:, 0], rand_vis[:, 1], c=y)
# plt.title('Random')

# accuracy - k
# plt.figure()
# ## pca
# plt.plot(np.arange(1, d+1), pca_acc, c='blue', label='PCA')
# ## random
# plt.plot(np.arange(1, d+1), rand_acc, c='red', label='random')
# plt.title('Accuracy and k')
# plt.xlabel('k')
# plt.ylabel('accuracy')

# singular value visualization
plt.figure()
plt.plot(np.arange(1, d+1), S)
plt.title('Singular value')
plt.legend()
plt.show()

