from __future__ import division
from __future__ import print_function
import numpy as np

'''
activation functions and their gradients
'''

def sigmoid(X):
    return 1.0 / (1 + np.exp(-X))

def tanh(X):
    a = np.exp(-2*X)
    return (1 - a) / (1 + a)

def relu(X):
    return X * (X > 0)

def sigmoid_grad(Y):
    return Y * (1 - Y)

def tanh_grad(Y):
    return 1 - Y ** 2

def relu_grad(Y):
    return (Y > 0).astype(float)

'''
def sigmoid_grad(X):
    Y = sigmoid(X)
    return Y * (1 - Y)

def tanh_grad(X):
    a = np.exp(X)
    return 4 / ((a + 1/a) ** 2)

def relu_grad(X):
    return (X > 0).astype(float)
'''


def cross_entropy(X_pred, X):
    '''
    input: X_pred: (0,1), X: {0,1}
    '''
    return - X * np.log(X_pred) - (1 - X) * np.log(1 - X_pred)

def shuffle_index(X):
    # shuffle the row indices for a matrix X
    return np.random.permutation(X.shape[0])
