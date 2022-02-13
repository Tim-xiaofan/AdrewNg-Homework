from sigmoid import sigmoid
import numpy as np
from gradient import gradient

'''
:parameter
theta:(n, 1)
X:(m, n)
y:(m,)
'''

def costFunction(theta, X, y):
    m = len(y)
    z = np.dot(X, theta)  # (m, n)  * (n, 1)
    p1 = np.dot(y.T, np.log(sigmoid(z)))  # (1, m) * (m, 1)
    p2 = np.dot(1 - y.T, np.log(1 - sigmoid(z)))
    J = -(1 / m) * np.sum(p1 + p2)
    return J, gradient(X, y, theta)
