from mnist import MNIST
import numpy as np
import kernel, kmm

if __name__ == '__main__':
    mndata = MNIST('../python-mnist/data/')
    xtr, ytr = mndata.load_training()
    xte, yte = mndata.load_testing()
    te_labels = [3, 4, 5]
    xtr = np.array(xtr)[0:200]
    ytr = np.array(ytr)[0:200]
    xte = np.array(xte)
    yte = np.array(yte)
    idxs = []
    for i in [0]:
        idxs.extend(np.where(yte == i)[0])
    xte = xte[idxs]
    yte = yte[idxs]
    res = kmm.kmm(xtr, xte, kernel.rbf, kfargs=(15, ), B=10)
    coef = np.array(res['x'])

    score = np.zeros([10, 1])
    for i in range(10):
        score[i] = np.mean(coef[np.where(ytr == i)])
