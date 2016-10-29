import numpy as np

def linear_gaussian(n_tr, n_te):
    """
    Generate training and testing data with covariate shift from a linear model f(x) = x + 1 + epsilon,
    where x_tr ~ N(1, 2) and x_te ~ N(0, 1), and epsilon is Gaussian noise.
    """
    x_tr = (np.random.randn(n_tr) + 1) * np.sqrt(2)
    x_te = (np.random.randn(n_te) + 0) * np.sqrt(1)
    y_tr = x_tr + 1 + np.random.randn(n_tr)
    y_te = x_te + 1 + np.random.randn(n_te)
    return x_tr, y_tr, x_te, y_te

def poly_gaussian(n_tr, n_te):
    """
    Generate training and testing data with covariate shift from a non-linear model f(x) = x^3 - x + 1 + epsilon,
    where x_tr ~ N(1, 2) and x_te ~ N(0, 1), and epsilon is Gaussian noise.
    """
    x_tr = (np.random.randn(n_tr) + 1) * np.sqrt(2)
    x_te = (np.random.randn(n_te) + 0) * np.sqrt(1)
    y_tr = x_tr ** 3 - x_tr + 1 + np.random.randn(n_tr)
    y_te = x_te ** 3 - x_te + 1 + np.random.randn(n_te)
    return x_tr, y_tr, x_te, y_te
