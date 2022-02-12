## Machine Learning Online Class - Exercise 1: Linear Regression#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exercise:
#
#     warmUpExercise.m
#     plotData.m
#     gradientDescent.m
#     computeCost.m
#     gradientDescentMulti.m
#     computeCostMulti.m
#     featureNormalize.m
#     normalEqn.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotData import plotData
from warmUpExercise import warmUpExercise
from computeCost import computeCost
from gradientDescent import gradientDescent
from gradientDescentMulti import gradientDescentMulti
import auxiliary
from visualizingJ import visualizingJ
from featureNormalize import featureNomalize

## ======================= Part 1: Feature Normalization =======================
np.set_printoptions(suppress=True)
data = pd.read_csv('ex1data2.txt', names=['square', 'bedrooms', 'price'])
data.head()
X, y = auxiliary.get_X_y(data)
# Scale features and set them to zero mean
print('Normalizing Features ...')
X_norm, mu, sigma = featureNomalize(X)

# Add intercept term to X
ones = pd.DataFrame({'ones': np.ones(len(X_norm))})
X = pd.concat([ones, X_norm], axis=1)
X = X.to_numpy()
y = y.to_numpy()
# print('After add ones:\n', X)

# ## =================== Part 2: Cost and Gradient descent ===================
#
# Some gradient descent settings
iterations = 400
alpha = 0.01

theta = np.zeros((X.shape[1], 1), dtype=np.double)
print('theta.shape:', theta.shape)
print('Running Gradient Descent ...')
# run gradient descent multi
theta, J_history = gradientDescentMulti(X, y, theta, alpha, iterations)
print('Theta found by gradient descent multi:', theta.T)
print('Expected theta values (approx): [334302.063993  99411.449474 3267.012854]')

# Plot the convergence graph
fig, ax = plt.subplots()
ax.plot(np.linspace(0, iterations, iterations), J_history, label=r'J(${\Theta})$')
ax.legend()
ax.set_ylabel(r'J(${\Theta}$)')
ax.set_xlabel('iters')
plt.show()

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.
sq_ft = 1650  # You should change this
br = 3
X = np.array([1, sq_ft, br])  # (3,)
price = np.dot(X, theta)
print('Estimate the price of a %d sq-ft, %d br house:' % (sq_ft, br), price)
