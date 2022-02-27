from sigmoid import sigmoid
import numpy as np


def gradient(X, y, theta):
    """
    计算逻辑回归的梯度

    Parameters
    ----------
    :param X: (m, n)
    :param y:  (m,)
    :param theta: (n, 1)
    :return:
    """
    m = len(X)
    z = np.dot(X, theta)  # (m, n)  * (n, 1)
    # print('z.shape:', z.shape)
    return (1 / m) * np.dot(X.T, (sigmoid(z) - y))
