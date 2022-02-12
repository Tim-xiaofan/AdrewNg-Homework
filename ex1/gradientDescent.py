import numpy as np
from computeCost import computeCost


def gradientDescent(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn `theta`. Updates theta by taking `num_iters`
    gradient steps with learning rate `alpha`.

    Parameters
    ----------
    X : array_like
        The input dataset of shape (m, n+1).

    y : array_like
        Value at given features. A vector of shape (m, ).

    theta : array_like
        Initial values for the linear regression parameters.
        A vector of shape (n+1, ).

    alpha : float
        The learning rate.

    num_iters : int
        The number of iterations for gradient descent.

    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).

    J_history : list
        A python list for the values of the cost function after each iteration.

    Instructions
    ------------
    Peform a single gradient step on the parameter vector theta.

    While debugging, it can be useful to print out the values of
    the cost function (computeCost) and gradient here.
    """
    m = len(X)
    theta_tmp = theta
    J_history = np.zeros((num_iters, 1))  # 存储每次迭代的代价函数值
    for i in range(num_iters):
        for j in range(len(theta)):  # 同时更新
            z = np.dot(X, theta) - y  # (m, 1) = (m, n) * (n * 1)
            xj = X[:, j]
            z = np.dot(z, xj[:, None])
            theta_tmp[j] = theta[j] - (alpha / m) * z
        # Save the cost J in every iteration
        J_history[i] = computeCost(X, y, theta)
        theta = theta_tmp
    return theta, J_history
