import numpy as np
import pandas as pd
from sigmoid import sigmoid

def predict(theta, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression.
    Computes the predictions for X using a threshold at 0.5
    (i.e., if sigmoid(theta.T*x) >= 0.5, predict 1)

    Parameters
    ----------
    theta : array_like
        Parameters for logistic regression. A vecotor of shape (n+1, ).

    X : array_like
        The data to use for computing predictions. The rows is the number
        of points to compute predictions, and columns is the number of
        features.

    Returns
    -------
    p : array_like
        Predictions and 0 or 1 for each row in X.

    Instructions
    ------------
    Complete the following code to make predictions using your learned
    logistic regression parameters.You should set p to a vector of 0's and 1's
    """
    m = X.shape[0]  # Number of training examples

    # You need to return the following variables correctly
    p = np.zeros(m)

    # ====================== YOUR CODE HERE ======================
    h = sigmoid(np.dot(X, theta))
    # print('h:\n', h)
    p = pd.DataFrame(h)
    # print('p1:\n', p)
    p[p.ge(0.5)] = 1
    p[p.le(0.5)] = 0
    # print('p2:\n', p)
    # ============================================================
    return p.to_numpy().ravel()
