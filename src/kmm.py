import numpy as np
from cvxopt import matrix, solvers

"""
Kernel Mean Matching
"""


def kmm(x_tr, x_te, kf=lambda (x,y,kfargs): np.dot(x, y), kfargs=None, B=1):
    """ Performs kernel mean matching on x_tr and x_te under the kernel function kf.
    This function uses cvxopt QP solver.
    Let fm be the feature mapping corresponding to kernel function kf, this functions returns a vector 'b' such that:

        b = argmin_w ||sum_{i=1}^n w_i fm(x_tr(i)) - 1 / m * sum_{j=1}^m fm(x_te(j))||^2
            s.t. w >= 0, sum(w_i) = 1

    Input:
        x_tr: np.ndarray of size n * p
        x_te: np.ndarray of size m * p
        kf: function handle; default is linear kernel
        kfargs: a sequence of arguments that has form (arg1, arg2, ..., argn). In the case of only 1 argument, use (arg1,)
        B: the constraint on the largest value an element of b can take

    Output:
        qp solver result

    Reference:
    Gretton et al, Covariate Shift by Kernel Mean Matching, 2009, http://www.gatsby.ucl.ac.uk/~gretton/papers/covariateShiftChapter.pdf
    """
    # TODO: Implement kmm with Frank-Wolfe
    ntr = len(x_tr)
    nte = len(x_te)
    epsilon = B / np.sqrt(ntr)
    K = matrix(kernel_matrix(x_tr, x_tr, kf, kfargs))
    kappa = matrix(ntr / nte * np.sum(kernel_matrix(x_tr, x_te, kf, kfargs), axis=1))
    G = matrix(np.r_[np.eye(ntr), -np.eye(ntr), np.ones([1, ntr]), -np.ones([1, ntr])])
    h = matrix(np.r_[np.ones([ntr, 1]) * B, np.zeros([ntr, 1]), np.array([[ntr * epsilon + ntr]]), np.array([[ntr * epsilon - ntr]])])

    res = solvers.qp(K, -kappa, G, h)
    return res

def kernel_matrix(x1, x2, kf, kfargs):
    """
    Build kernel matrix
    """
    l1 = len(x1)
    l2 = len(x2)
    K = np.zeros([l1, l2])
    for i in range(l1):
        for j in range(l2):
            #if (j < i):
            #    K[i][j] = K[j][i]
            #else:
            K[i][j] = kf(x1[i], x2[j], *kfargs)
    return K
