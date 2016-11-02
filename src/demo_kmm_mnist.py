from mnist import MNIST
import numpy as np
import kernel, kmm

def kmm_train(test_labels, kf, kfargs, B):
    idx_te = list()
    for i in test_labels:
        idx_te.extend(np.where(yte == i)[0])
    print len(idx_te)
    res = kmm.kmm(xtr, xte, kf, kfargs, B)
    coef = np.array(res['x'])
    return coef

if __name__ == '__main__':
    test_labels = [3, 4, 5] # Define labels in test set
    tr_p = 0.01 # Proportion of training data subsampled for compuational simplicity

    mndata = MNIST('../python-mnist/data/')
    xtr, ytr = mndata.load_training()
    xte, yte = mndata.load_testing()
    idx_tr = np.where(np.random.rand(len(ytr), 1) < tr_p)[0]
    [xtr, ytr] = [np.array(xtr)[idx_tr], np.array(ytr)[idx_tr]]
    [xte, yte] = [np.array(xte), np.array(yte)]

    coef = kmm_train(test_labels, kernel.rbf, (15, ), 10)

    score = np.zeros([10, 1])
    for i in range(10):
        score[i] = np.mean(coef[np.where(ytr == i)])
