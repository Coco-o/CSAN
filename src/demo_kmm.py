import kernel, kmm, gen_synthetic
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    [xtr, ytr, xte, yte] = gen_synthetic.poly_gaussian(100, 50)
    res = kmm.kmm(xtr, xte, kernel.rbf, kfargs=(1, ), B=10)
    coef = np.array(res['x'])
    plt.figure()
    plt.hold(True)
    plt.scatter(xtr, ytr, color='black', marker='x')
    plt.scatter(xte, yte, color='red')
    plt.scatter(xtr, ytr, color='green', s=coef*100, alpha=0.5)
    plt.show()
