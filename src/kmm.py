import numpy as np

"""
Kernel Mean Matching
"""


def kmm(x_tr, x_te, fm=lambda x:x):
    """ Performs kernel mean matching on x_tr and x_te under the feature mapping function fm.
    Returns a vector 'b' such that:

        b = argmin_w ||sum_{i=1}^n w_i fm(x_tr(i)) - 1 / m * sum_{j=1}^m fm(x_te(j))||^2
            s.t. w >= 0, sum(w_i) = 1

    :type x_tr: np.ndarray
    :type x_te: np.ndarray
    """
    # TODO: Implement kmm with Frank-Wolfe
