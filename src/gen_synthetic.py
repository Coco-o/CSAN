import numpy as np

def linear_gaussian(n_tr, n_te):
    """
    Generate training and testing data with covariate shift from a linear model f(x) = x + 1 + epsilon,
    where x_tr ~ N(0.5, 0.25) and x_te ~ N(0, 0.09), and epsilon ~ N(0, 0.01)
    """
    x_tr = (np.random.randn(n_tr) + 0.5) * 0.5
    x_te = (np.random.randn(n_te) + 0) * 0.3
    y_tr = x_tr + 1 + np.random.randn(n_tr) * 0.1
    y_te = x_te + 1 + np.random.randn(n_te) * 0.1
    return x_tr, y_tr, x_te, y_te

def poly_gaussian(n_tr, n_te):
    """
    Generate training and testing data with covariate shift from a non-linear model f(x) = x^3 - x + 1 + epsilon,
    where x_tr ~ N(0.5, 0.25) and x_te ~ N(0, 0.09), and epsilon ~ N(0, 0.01)
    """
    x_tr = (np.random.randn(n_tr) + 0.5) * 0.5
    x_te = (np.random.randn(n_te) + 0) * 0.3
    y_tr = x_tr ** 3 - x_tr + 1 + np.random.randn(n_tr) * 0.1
    y_te = x_te ** 3 - x_te + 1 + np.random.randn(n_te) * 0.1
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
