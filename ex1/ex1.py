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

## ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.m
print('Running warmUpExercise ... ')
print('5x5 Identity Matrix: ')
print(warmUpExercise())

print('Program paused. Press enter to continue.')
# input() 太麻烦的

## ======================= Part 2: Plotting =======================
print('Plotting Data ...')
data = pd.read_csv('ex1data1.txt', names=['population', 'profit'])  # 读取数据并赋予列名
m = len(data)  # number of training examples

# Plot Data
# Note: You have to complete the code in plotData.m
plotData(data['population'], data['profit'])

print('Program paused. Press enter to continue.')
# input()

## =================== Part 3: Cost and Gradient descent ===================

# Add a column of ones to x
ones = pd.DataFrame({'ones': np.ones(len(data))})
data = pd.concat([ones, data], axis=1)
# print("After concat ones, data:\n", data)

# Some gradient descent settings
iterations = 1500
alpha = 0.01

print('Testing the cost function ...')
# 提取特征量X和输出y
X, y = auxiliary.get_X_y(data)
X = X.to_numpy()
y = y.to_numpy()
print('X.shape:', X.shape)
print('Y.shape:', y.shape)
# compute and display initial cost
theta = np.array([0, 0], dtype=np.double)
print('theta.shape:', theta.shape)
J = computeCost(X, y, theta)
print('With theta = [0 , 0],Cost computed = ', J)
print('Expected cost value (approx) 32.07')

# further testing of the cost function
J = computeCost(X, y, np.array([-1, 2]))
print('With theta = [-1 , 2], Cost computed = ', J)
print('Expected cost value (approx) 54.24')

print('Program paused. Press enter to continue.')
# input()

print('Running Gradient Descent ...')
# run gradient descent
theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
print('Theta found by gradient descent:', theta)
theta = np.array([0, 0], dtype=np.double)
# theta = gradientDescentMulti(X, y, theta, alpha, iterations)
# print('Theta found by gradient descent multi:', theta)
print('Expected theta values (approx): [-3.6303  1.1664]')

plotData(data['population'], data['profit'])
plt.plot(X[:, 1], np.dot(X, theta))
plt.legend(['Training Data', 'Linear regression'])
plt.show()

print('Plotting J_history...')
fig, ax = plt.subplots()
ax.plot(np.linspace(0, iterations, iterations), J_history, label=r'J(${\Theta})$')
ax.legend()
ax.set_ylabel(r'J(${\Theta}$)')
ax.set_xlabel('iters')
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot(np.array([[1, 3.5]]), theta)  # (1,2)*(2,1)
print('For population = 35,000, we predict a profit of ', predict1 * 10000)
predict2 = np.dot(np.array([[1, 7]]), theta)
print('For population = 70,000, we predict a profit of ', predict2 * 10000)
print('Program paused. Press enter to continue.\n')
# input()

# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...')
visualizingJ(X, y, theta)
