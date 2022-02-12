import numpy as np


def featureNomalize(X):
    """
    常用feature scaling方法--Standardization (Z-score Normalization)
    x_norm = (x - mean(x)) / std_deviation(x)
    Normalizes the features in X. returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when working with
    learning algorithms.

    Parameters
    ----------
    X : DataFrame
        The dataset of shape (m, n).

    Returns
    -------
    X_norm : DataFrame
        The normalized dataset of shape (m, n).

    Instructions
    ------------
    First, for each feature dimension, compute the mean of the feature
    and subtract it from the dataset, storing the mean value in mu.
    Next, compute the  standard deviation of each feature and divide
    each feature by it's standard deviation, storing the standard deviation
    in sigma.

    Note that X is a matrix where each column is a feature and each row is
    an example. You needto perform the normalization separately for each feature.

    Hint
    ----
    You might find the 'np.mean' and 'np.std' functions useful.
    """
    # You need to set these values correctly
    X_norm = X.copy()  # 特征缩放结果
    mu = np.zeros(X.shape[-1])  # 均值
    sigma = np.zeros(X.shape[-1])  # 标准差

    # =========================== YOUR CODE HERE =====================
    mu = X.mean(axis=0)
    print('mean(X):', mu.head())
    sigma = X.std(axis=0, ddof=0)  # 按列分，除以n
    print('sigma(X):', sigma.head())
    X_norm = X.apply(lambda column: (column - column.mean()) / column.std(ddof=0))  # 除以n
    # ================================================================
    return X_norm, mu, sigma
