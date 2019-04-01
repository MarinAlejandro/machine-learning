import numpy as np


def polynomial_kernel(x_i, x_j, degree=2, *others):
    k = np.power(x_i.T @ x_j + 1, degree)
    return k


def gauss_kernel(x_i, x_j, gamma=100, *others):
    k = (x_i - x_j).T @ (x_i - x_j)
    k = np.exp(-gamma * k)
    return k
