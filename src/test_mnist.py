from __future__ import division
from __future__ import print_function
import numpy as np
from util import *
from classifiers import WeightedBinaryNN
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from model2 import AdvSampler, WeightedLR
import kernel, kmm
import matplotlib.gridspec as gridspec
from mnist import MNIST

dim = 784
ns, nt = 500, 500 # samples per digit
s_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
t_digits = [5]
ks = len(s_digits)
kt = len(t_digits)

WRITE_DATA = True
DO_KMM = True
name = '5'


if WRITE_DATA:
    # load data, 50000, 10000, 10000
    data_path = '../python-mnist/'
    mndata = MNIST(data_path)
    x_train, y_train = mndata.load_training()
    x_test, y_test = mndata.load_testing()
    x_train = np.array(x_train) / 255.0
    y_train = np.array(y_train)
    x_test = np.array(x_test) / 255.0
    y_test = np.array(y_test)

    #x_train = np.genfromtxt(data_path+'train_full.images', delimiter=' ') / 255.0
    #x_test = np.genfromtxt(data_path+'test_full.images', delimiter=' ') / 255.0
    #y_train = np.genfromtxt(data_path+'train_full.labels', delimiter=' ').astype(int)
    #y_test = np.genfromtxt(data_path+'test_full.labels', delimiter=' ').astype(int)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
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
    print('finished')

    #sample data:


    #get xs xt ys yt
    xs = np.zeros((ns*ks, dim))
    ys = np.zeros(ns*ks).astype(int)
    xt = np.zeros((nt*kt, dim))
    yt = np.ones(nt*kt).astype(int)
    for i in range(ks):
        d = s_digits[i]
        xs[ns*i:ns*(i+1),:] = train_collection[d][:ns,:]
        ys[ns*i:ns*(i+1)] = int(d in t_digits)

    for i in range(kt):
        d = t_digits[i]
        xt[nt*i:nt*(i+1),:] = test_collection[d][:nt,:]
        yt[nt*i:nt*(i+1)] = 1

    #write data
    np.savetxt('../data/xs'+name+'.csv', xs, delimiter=',', fmt='%.4f')
    np.savetxt('../data/ys'+name+'.csv', ys, delimiter=',', fmt='%d')
    np.savetxt('../data/xt'+name+'.csv', xt, delimiter=',', fmt='%.4f')
    np.savetxt('../data/yt'+name+'.csv', yt, delimiter=',', fmt='%d')




print('Loading data ...',)

# load saved data:
xs = np.genfromtxt('../data/xs'+name+'.csv', delimiter=',')
ys = np.genfromtxt('../data/ys'+name+'.csv', delimiter=',')
xt = np.genfromtxt('../data/xt'+name+'.csv', delimiter=',')
yt = np.genfromtxt('../data/yt'+name+'.csv', delimiter=',')
print(xs.shape, xt.shape, ys.shape, yt.shape)







parameter1 = {
    'layers': [dim, 200, 100, 1],
    'activation': 'relu',
    'weight_decay': 0.0001
}
parameter2 = {
    'layers': [dim, 200, 100, 1],
    'activation': 'relu',
    'weight_decay': 0.0001
}
parameter3 = {
    'layers': [dim, 200, 100, 1],
    'activation': 'relu',
    'weight_decay': 0.0001
}
model_parameter = {
    'sampler_parameter': parameter1,
    'adv_acc_parameter': parameter2,
    'adv_rej_parameter': parameter3,
    'adv_acc_step_num': 1, #if using sgd, set this smaller
    'adv_acc_learning_rate': 0.001,
    'adv_rej_step_num': 1, #if using sgd, set this smaller
    'adv_rej_learning_rate': 0.001,
    'sampler_step_num': 1,
    'sampler_learning_rate': 0.001,
    'coeff': 1.0,
    'sgd': True,
    'sampler_batch_size': 1.0/xs.shape[0], #proportion of data_size, 1 means batch training
    'update_pred_freq': 1 # how many times to update self.pred_s in 1 epoch
}


if __name__ == '__main__':

    np.random.seed(0)

    step_num = 10

    if DO_KMM:
        res3 = np.array(kmm.kmm(xs, xt, kernel.polykernel, kfargs=(1,2 ), B=10)['x']).reshape((ns*10, 1))
        for i in range(10):
            print(np.sum(res3[i*ns:i*ns+ns,:]>1)/ns)
        np.savetxt('../data/kmm_res'+name+'.csv', res3, delimiter=',', fmt='%.4f')

    sampler = AdvSampler(model_parameter)
    log = sampler.train(xs, xt, step_num)

    res = sampler.get_result(xs, 'sampler')
    for i in range(10):
        print(np.sum(res[i*ns:i*ns+ns,:]>0.5)/ns)
        ys[i*ns:i*ns+ns] = i
        np.savetxt('../data/gan_res'+name+'.csv', res, delimiter=',', fmt='%.4f')

    plt.scatter(ys, res)
    plt.savefig('figs/mnist.png')
    plt.show()

    '''
    model_parameter['coeff'] = 0.0
    sampler = AdvSampler(model_parameter)
    sampler.train(x_tr, x_te, step_num)
    res2 = sampler.get_result(x_tr, 'sampler')
    lr = WeightedLR(dim)
    lr.train(x_tr, y_tr, res2, lr_rate)
    lr_loss = lr.get_loss(x_te, y_te, np.ones_like(y_te))
    print(lr_loss)
    plot_weights(x_tr, y_tr, x_te, y_te, res2.flatten(), lr, 'figs/syn2.pdf')
    '''
