import kernel, kmm, gen_synthetic
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    np.random.seed(0)
    [xtr, ytr, xte, yte] = gen_synthetic.poly_uniform(700, 200)
    res = kmm.kmm(xtr, xte, kernel.rbf, kfargs=(1, ), B=10)
    coef = np.array(res['x'])
    plt.figure()
    plt.hold(True)
    plt.scatter(xtr, ytr, color='black', marker='x')
    plt.scatter(xte, yte, color='red')
    plt.scatter(xtr, ytr, color='green', s=coef*100, alpha=0.5)
    ylim = plt.gca().get_ylim()
    plt.vlines(xtr, ylim[0], coef.flatten()/coef.max() + ylim[0], color='m')
    plt.gca().set_ylim(ylim)
    plt.legend(('training', 'testing', 'weighted training', 'weights'))
    plt.show()
