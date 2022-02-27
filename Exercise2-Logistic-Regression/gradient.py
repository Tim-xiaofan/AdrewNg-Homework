from sigmoid import sigmoid
import numpy as np

# 计算梯度步长
'''
:parameter
theta:(n, 1)
X:(m, n)
y:(m,)
'''
def gradient(X, y, theta):
    m = len(X)
    z = np.dot(X, theta)  # (m, n)  * (n, 1)
    # print('z.shape:', z.shape)
    return (1 / m) * np.dot(X.T, (sigmoid(z) - y))

