import numpy as np


# 向量换实现的代价函数计算
# X:(m, n)
# theta:(n, 1)
def computeCostMulti(X, y, theta):
    m = len(X)
    z = np.dot(X, theta) - y[:, None]
    return np.sum(np.square(z)) / (2 * m)
