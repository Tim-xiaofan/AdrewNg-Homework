# 辅助函数
import numpy as np


def sigmoid(z):  # s型函数
    return 1 / (1 + np.exp(-z))
