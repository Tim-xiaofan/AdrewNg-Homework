import numpy as np
from computeCostMulti import computeCostMulti

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = len(X)
    theta_tmp = theta  # (n, 1)
    J_history = np.zeros((num_iters, 1))
    # print('theta.shape:', theta.shape)
    # print('X.shape:', X.shape)
    # print('y.shape:', y.shape)
    for i in range(num_iters):
        z = np.dot(X, theta)  # (m, 1) = (m, n) * (n, 1)
        # print('z.shape0:', z.shape)
        z = z - y[:, None]
        # print('z.shape1:', z.shape)
        z = np.dot(X.T, z)  # (n, 1) = (n, m) * (m, 1)
        # print('z.shape2:', z.shape)
        theta_tmp = theta - (alpha / m) * z
        # Save the cost J in every iteration
        J_history[i] = computeCostMulti(X, y, theta)
        theta = theta_tmp
    return theta, J_history
