# 辅助函数
import numpy as np

def get_X_y(data):
    return [data.iloc[:, :-1], data.iloc[:, -1]]

