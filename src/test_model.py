from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib # comment these two lines if running remotely
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model2 import AdvSampler, WeightedLR
import kernel, kmm
import matplotlib.gridspec as gridspec
from mnist import MNIST

#print(':')

def plot_log(log, filename):
    result = np.array(log)
    epoch = result[:,0]
    tloss = result[:,1]
    vloss = result[:,2]
    tl, = plt.plot(epoch, tloss, ls = '-', lw = 3)
    vl, = plt.plot(epoch, vloss, ls = '--', lw = 3)
    plt.legend([tl, vl], ['adversary', 'collaborator'])
    plt.xlabel('training step')
    plt.ylabel('cross entropy loss')
    if filename != '':
        plt.savefig(filename)
    else:
        plt.show()
    plt.clf()

def plot_weights(xtr, ytr, xte, yte, coef, lr, filename):
    gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax1.hold(True)
    ax1.scatter(xtr, ytr, color='black', marker='x')
    ax1.scatter(xte, yte, color='red')
    ax1.legend(('source', 'target'))
    #ax1.plot(xtr, lr.predict(xtr), color='blue')
    xlim = ax1.get_xlim()
    ax2.vlines(xtr, 0, coef.flatten(), color='m')
    ax2.set_xlim(xlim)
    ax2.legend(('weights',))
    if filename != '':
        plt.savefig(filename)
    else:
        plt.show()
    plt.clf()

def poly_gaussian(n_tr, n_te):
    """
    Generate training and testing data with covariate shift from a non-linear model f(x) = x^3 - x + 1 + epsilon,
    where x_tr ~ N(0.5, 0.25) and x_te ~ N(0, 0.09), and epsilon ~ N(0, 0.01)
    """
    x_tr = (np.random.randn(n_tr, 1) + 0.5) * 0.5
    x_te = (np.random.randn(n_te, 1) + 0.0) * 0.3
    y_tr = x_tr ** 3 - x_tr + 1 + np.random.randn(n_tr, 1) * 0.1
    y_te = x_te ** 3 - x_te + 1 + np.random.randn(n_te, 1) * 0.1
    return x_tr, y_tr, x_te, y_te

def poly_uniform(n_tr, n_te):
    x_tr = np.random.uniform(-1.5, 1.5, (n_tr, 1))
    x_te = np.random.uniform(-0.5, 0.5, (n_te, 1))
    y_tr = x_tr ** 3 - x_tr + 1 + np.random.randn(n_tr, 1) * 0.1
    y_te = x_te ** 3 - x_te + 1 + np.random.randn(n_te, 1) * 0.1
    return x_tr, y_tr, x_te, y_te

dim = 1
parameter1 = {
    'layers': [dim, 16, 6, 1],
    'activation': 'relu',
    'weight_decay': 0
}
parameter2 = {
    'layers': [dim, 16, 6, 1],
    'activation': 'relu',
    'weight_decay': 0
}
parameter3 = {
    'layers': [dim, 16, 6, 1],
    'activation': 'relu',
    'weight_decay': 0
}
model_parameter = {
    'sampler_parameter': parameter1,
    'adv_acc_parameter': parameter2,
    'adv_rej_parameter': parameter3,
    'adv_acc_step_num': 10, #if using sgd, set this smaller
    'adv_acc_learning_rate': 0.1,
    'adv_rej_step_num': 10, #if using sgd, set this smaller
    'adv_rej_learning_rate': 0.1,
    'sampler_step_num': 20,
    'sampler_learning_rate': 0.1,
    'coeff': 1.0,
    'sgd': True,
    'sampler_batch_size': 0.1, #proportion of data_size, 1 means batch training
    'update_pred_freq': 1 # how many times to update self.pred_s in 1 epoch
}


if __name__ == '__main__':

    np.random.seed(0)
    x_tr, y_tr, x_te, y_te = poly_uniform(300, 100)
    step_num = 25
    lr_rate = 0.1

    sampler = AdvSampler(model_parameter)
    log = sampler.train(x_tr, x_te, step_num)
    plot_log(log, 'figs/syn_loss.pdf')
    res1 = sampler.get_result(x_tr, 'sampler')
    lr = WeightedLR(dim)
    lr.train(x_tr, y_tr, res1, lr_rate)
    lr_loss = lr.get_loss(x_te, y_te, np.ones_like(y_te))
    print(lr_loss)
    plot_weights(x_tr, y_tr, x_te, y_te, res1.flatten(), lr, 'figs/syn1.pdf')



    model_parameter['coeff'] = 0.0
    sampler = AdvSampler(model_parameter)
    sampler.train(x_tr, x_te, step_num)
    res2 = sampler.get_result(x_tr, 'sampler')
    lr = WeightedLR(dim)
    lr.train(x_tr, y_tr, res2, lr_rate)
    lr_loss = lr.get_loss(x_te, y_te, np.ones_like(y_te))
    print(lr_loss)
    plot_weights(x_tr, y_tr, x_te, y_te, res2.flatten(), lr, 'figs/syn2.pdf')



    res3 = kmm.kmm(x_tr, x_te, kernel.rbf, kfargs=(1, ), B=10)
    coef = np.array(res3['x'])
    lr = WeightedLR(dim)
    lr.train(x_tr, y_tr, coef.reshape((coef.shape[0], 1)), lr_rate)
    lr_loss = lr.get_loss(x_te, y_te, np.ones_like(y_te))
    print(lr_loss)
    plot_weights(x_tr, y_tr, x_te, y_te, coef, lr, 'figs/syn3.pdf')
