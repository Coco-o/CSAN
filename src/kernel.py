import numpy as np

def polykernel(x1, x2, offset=0, degree=2):
    """
    polynomial kernel of the form:
    k(x1, x2) = ((x1' * x2) + offset) ^ degree
    """
    return (np.dot(x1, x2) + offset)**degree

def rbf(x1, x2, sigma=1):
    """
    rbf kernel. sigma is the smoothing parameter
    k(x1, x2) = exp(-||x1 - x2||^2 / 2 * sigma^2)
    """
    return np.exp(-(np.linalg.norm(x1 - x2) / (2 * sigma**2)))
