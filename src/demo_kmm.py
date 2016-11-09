import kernel, kmm, gen_synthetic
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

if __name__ == '__main__':
    np.random.seed(0)
    [xtr, ytr, xte, yte] = gen_synthetic.poly_uniform(700, 200)
    res = kmm.kmm(xtr, xte, kernel.rbf, kfargs=(1, ), B=10)
    coef = np.array(res['x'])
    gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax1.hold(True)
    ax1.scatter(xtr, ytr, color='black', marker='x')
    ax1.scatter(xte, yte, color='red')
    ax1.scatter(xtr, ytr, color='green', s=coef*100, alpha=0.5)
    ax1.legend(('training', 'testing', 'weighted training'))
    xlim = ax1.get_xlim()
    ax2.vlines(xtr, 0, coef.flatten()/coef.max(), color='m')
    ax2.set_xlim(xlim)
    ax2.legend(('weights',))
    plt.show()
