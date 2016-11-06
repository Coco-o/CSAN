from __future__ import division
from __future__ import print_function
import numpy as np
from util import *
from classifiers import WeightedBinaryNN
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

'''
# load data, a smaller dataset 3000, 1000, 3000

data_path = '../datasets/mnist/'
train = np.genfromtxt(data_path+'train_small.txt', delimiter = ',')
valid = np.genfromtxt(data_path+'valid_small.txt', delimiter = ',')
test = np.genfromtxt(data_path+'test_small.txt', delimiter = ',')
x_train = train[:, :-1]
y_train = train[:, -1].astype(int)
x_valid = valid[:, :-1]
y_valid = valid[:, -1].astype(int)
x_test = test[:, :-1]
y_test = test[:, -1].astype(int)
'''

# load data, 50000, 10000, 10000
data_path = '../datasets/mnist/'
x_train = np.genfromtxt(data_path+'train_full.images', delimiter=' ')/255
y_train = np.genfromtxt(data_path+'train_full.labels', delimiter=' ').astype(int)
x_test = np.genfromtxt(data_path+'test_full.images', delimiter=' ')/255
y_test = np.genfromtxt(data_path+'test_full.labels', delimiter=' ').astype(int)
idx = np.random.permutation(x_train.shape[0])
x_train_old, y_train_old = x_train[idx, :], y_train[idx]
valid_size = 10000
train_size = x_train.shape[0] - valid_size
x_train, y_train, x_valid, y_valid = x_train_old[:train_size, :], y_train_old[:train_size], \
    x_train_old[train_size:, :], y_train_old[train_size:]



#split datasets according to labels:
train_collection, valid_collection, test_collection = [], [], []
for i in range(10):
    x = x_train[(y_train == i), :]
    train_collection.append(x)
    x = x_valid[(y_valid == i), :]
    valid_collection.append(x)
    x = x_test[(y_test == i), :]
    test_collection.append(x)
    '''
    print(x.shape[0])
    j = np.random.randint(0,x.shape[0])
    plt.imshow(np.reshape(x[j,:],(28,28)), cmap='gray', interpolation='none')
    plt.show()
    '''



'''
train digit_0 vs digit_1
'''
digit_0 = 7
digit_1 = 9
x_train = np.vstack((train_collection[digit_0], train_collection[digit_1]))
y_0, y_1 = np.zeros((train_collection[digit_0].shape[0], 1)), np.ones((train_collection[digit_1].shape[0], 1))
y_train = np.vstack((y_0, y_1)).astype(int)
x_valid = np.vstack((valid_collection[digit_0], valid_collection[digit_1]))
y_0, y_1 = np.zeros((valid_collection[digit_0].shape[0], 1)), np.ones((valid_collection[digit_1].shape[0], 1))
y_valid = np.vstack((y_0, y_1)).astype(int)


# shuffle train:
idx = shuffle_index(x_train)
x_train = x_train[idx, :]
y_train = y_train[idx, :]

'''
for i in range(10):
    j = np.random.randint(0, x_train.shape[0])
    print(y_train[j, 0])
    plt.imshow(np.reshape(x_train[j,:],(28,28)), cmap='gray', interpolation='none')
    plt.show()
'''


dim = x_train.shape[1]
train_size = x_train.shape[0]
parameters = {
    'layers': [dim, 100, 1],
    'activation': 'relu',
    'weight_decay': 0
}

np.random.seed(0)
model = WeightedBinaryNN(parameters)
model.random_init()

'''batch training'''
step_num = 1000
learning_rate = 0.1
importance = np.ones_like(y_train)
for i in range(step_num):
    model.train_batch(x_train, y_train, importance, learning_rate)
    pred_prob_tr = model.predict(x_train)
    pred_prob_vl = model.predict(x_valid)
    acc_tr = np.sum((pred_prob_tr>0.5).astype(int) == y_train) / y_train.shape[0]
    loss_tr = np.sum(cross_entropy(pred_prob_tr, y_train)) / y_train.shape[0]
    acc_vl = np.sum((pred_prob_vl>0.5).astype(int) == y_valid) / y_valid.shape[0]
    loss_vl = np.sum(cross_entropy(pred_prob_vl, y_valid)) / y_valid.shape[0]
    print(i, acc_tr, acc_vl, loss_tr, loss_vl)

'''SGD training'''
