from __future__ import division
from __future__ import print_function
import numpy as np
from model import AdvSampler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gen_synthetic

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
    """
    Generate training and testing data with covariate shift from a non-linear model f(x) = x^3 - x + 1 + epsilon,
    where x_tr ~ Uniform(-1.5, 1.5) and x_te ~ Uniform(-0.5, 0.5), and epsilon ~ N(0, 0.01)
    """
    x_tr = np.random.uniform(-1.5, 1.5, (n_tr, 1))
    x_te = np.random.uniform(-0.5, 0.5, (n_te, 1))
    y_tr = x_tr ** 3 - x_tr + 1 + np.random.randn(n_tr, 1) * 0.1
    y_te = x_te ** 3 - x_te + 1 + np.random.randn(n_te, 1) * 0.1
    return x_tr, y_tr, x_te, y_te

dim = 1
parameter1 = {
    'layers': [dim, 16, 6, 1],
    'activation': 'relu',
    'weight_decay': 0.001
}
parameter2 = {
    'layers': [dim, 16, 6, 1],
    'activation': 'relu',
    'weight_decay': 0.001
}
model_parameter = {
    'sampler_parameter': parameter1,
    'adv_acc_parameter': parameter2,
    'adv_rej_parameter': parameter2,
    'adv_step_num': 5, #if using sgd, set this smaller
    'sampler_step_num': 50,
    'adv_learning_rate': 0.1,
    'sampler_learning_rate': 0.1,
    'coeff': 1.0,
    'sgd': True
}


if __name__ == '__main__':
    np.random.seed(0)
    x_tr, y_tr, x_te, y_te = gen_synthetic.poly_uniform(700, 200)
    sampler = AdvSampler(model_parameter)
    step_num = 100
    sampler.train(x_tr, x_te, step_num)

    res = sampler.get_result(x_tr, 'sampler')

    gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax1.hold(True)
    ax1.scatter(x_tr, y_tr, color='black', marker='x')
    ax1.scatter(x_te, y_te, color='red')
    ax1.scatter(x_tr, y_tr, color='green', s=res.flatten()*100, alpha=0.5)
    ax1.legend(('training', 'testing', 'weighted training'))
    xlim = ax1.get_xlim()
    ax2.vlines(x_tr, 0, res.flatten()/res.max(), color='m')
    ax2.set_xlim(xlim)
    ax2.legend(('weights',), loc='best')

    plt.show()
    plt.clf()


    '''
    res = sampler.get_result(x_tr, 'adv_acc')
    #plt.figure()
    #plt.hold(True)
    plt.scatter(x_tr, y_tr, color='black', marker='x')
    plt.scatter(x_te, y_te, color='red')
    plt.scatter(x_tr, y_tr, color='green', s=res.flatten()*100, alpha=0.5)
    plt.show()

    res = sampler.get_result(x_tr, 'adv_rej')
    #plt.figure()
    #plt.hold(True)
    plt.scatter(x_tr, y_tr, color='black', marker='x')
    plt.scatter(x_te, y_te, color='red')
    plt.scatter(x_tr, y_tr, color='green', s=res.flatten()*100, alpha=0.5)
    plt.show()
    '''
