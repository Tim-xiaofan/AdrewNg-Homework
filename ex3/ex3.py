# used for manipulating directory paths
import os
# Scientific and vector computation for python
import numpy as np
# Plotting library
import matplotlib.pyplot as plt
# will be used to load MATLAB mat datafile format
from scipy.io import loadmat
# library written for this exercise providing additional functions for assignment submission, and others
import utils
from lrCostFunction import lrCostFunction
from oneVsAll import *
from predictOneVsAll import *

# define the submission/grader object for this exercise
grader = utils.Grader()

# # 20x20 Input Images of Digits
# input_layer_size = 400
#
# # 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)
# num_labels = 10
#
# #  training data stored in arrays X, y
# data = loadmat(os.path.join('Data', 'ex3data1.mat'))
# X, y = data['X'], data['y'].ravel()
#
# # set the zero digit to 0, rather than its mapped 10 in this dataset
# # This is an artifact due to the fact that this dataset was used in
# # MATLAB where there is no index 0
# y[y == 10] = 0
#
# m = y.size
#
# print('\n\n===========1.2 Visualizing the data==========')
# # Randomly select 100 data points to display
# rand_indices = np.random.choice(m, 100, replace=False)
# sel = X[rand_indices, :]
#
# utils.displayData(sel)
# plt.show()
#
# print('\n\n===========1.3 Vectorizing Logistic Regression==========')
# # test values for the parameters theta
# theta_t = np.array([-2, -1, 1, 2], dtype=float)
# # test values for the inputs
# X_t = np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F') / 10.0], axis=1)
# # test values for the labels
# y_t = np.array([1, 0, 1, 0, 1])
# # test value for the regularization parameter
# lambda_t = 3
# J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)
# print('Cost         : {:.6f}'.format(J))
# print('Expected cost: 2.534819')
# print('-----------------------')
# print('Gradients:')
# print(' [{:.6f}, {:.6f}, {:.6f}, {:.6f}]'.format(*grad))
# print('Expected gradients:')
# print(' [0.146561, -0.548558, 0.724722, 1.398003]')
# # appends the implemented function in part 1 to the grader object
# grader[1] = lrCostFunction
#
# print('\n\n===========1.4 One-vs-all Classification==========')
# lambda_ = 0.1
# all_theta = oneVsAll(X, y, num_labels, lambda_)
# grader[2] = oneVsAll
# pred = predictOneVsAll(all_theta, X)
# print('Training Set Accuracy: {:.2f}%'.format(np.mean(pred == y) * 100))
# grader[3] = predictOneVsAll
# # grader.grade()

print('\n\n===========2 Neural Networks===========')
#  training data stored in arrays X, y
data = loadmat(os.path.join('Data', 'ex3data1.mat'))
X, y = data['X'], data['y'].ravel()

# set the zero digit to 0, rather than its mapped 10 in this dataset
# This is an artifact due to the fact that this dataset was used in
# MATLAB where there is no index 0
y[y == 10] = 0

# get number of examples in dataset
m = y.size

# randomly permute examples, to be used for visualizing one
# picture at a time
indices = np.random.permutation(m)

# Randomly select 100 data points to display
rand_indices = np.random.choice(m, 100, replace=False)
sel = X[rand_indices, :]

utils.displayData(sel)
print('\n\n===========2 Neural Networks===========')
