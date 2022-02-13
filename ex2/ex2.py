# used for manipulating directory paths
import os
# Scientific and vector computation for python
import numpy as np
# Plotting library
from matplotlib import pyplot as plt
# Optimization module in scipy
from scipy import optimize
# pandas
import pandas as pd
# library written for this exercise providing additional functions for assignment submission, and others
import utils
# define the submission/grader object for this exercise
from plot_data import plot_data
import auxiliary
from costFunction import costFunction
from sigmoid import sigmoid
from predict import predict
from ex2_reg import ex2_reg

grader = utils.Grader()

# Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.

data = pd.read_csv('./data/ex2data1.txt', names=['Exam 1', 'Exam 2', 'Admitted'])
# data.info()
print('data, first 5 lines:\n', data.iloc[0:5, :])
X, y = auxiliary.get_X_y(data)

## ==================== Part 1: Plotting ====================
# We start the exercise by first plotting the data to understand the
# the problem we are working with.
print('\n\n==================== Part 1: Plotting ====================')
plot_data(X, y)
plt.savefig('./Figure/dataset1.png')

# appends the implemented function in part 1 to the grader object
grader[1] = sigmoid

## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in
#  costFunction.m
print('\n\n============ Part 2: Compute Cost and Gradient ============')
#  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = [X.shape[0], X.shape[1]]

# Add intercept term to x and X_test
ones = pd.DataFrame({'ones': np.ones(m)})
X = pd.concat([ones, X], axis=1)

# Initialize fitting parameters
theta = np.zeros((n + 1), dtype=np.double)

# Compute and display initial cost and gradient
cost, grad = costFunction(theta, X, y)
print('Cost at initial theta (zeros):', cost)
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros): ', grad)
print('Expected gradients (approx): -0.1000 -12.0092 -11.2628')
grader[2] = costFunction
grader[3] = costFunction

# % ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.
print('\n\n============= Part 3: Optimizing using fminunc  =============')
# set options for optimize.minimize
options = {'maxiter': 400}

# see documention for scipy's optimize.minimize  for description about
# the different parameters
# The function returns an object `OptimizeResult`
# We use truncated Newton algorithm for optimization which is
# equivalent to MATLAB's fminunc
# See https://stackoverflow.com/questions/18801002/fminunc-alternate-in-numpy
res = optimize.minimize(fun=costFunction,
                        x0=theta,
                        args=(X, y),
                        jac=True,
                        method='TNC',
                        options=options)

# the fun property of `OptimizeResult` object returns
# the value of costFunction at optimized theta
cost = res.fun

# the optimized theta is in the x property
theta = res.x

# Print theta to screen
print('Cost at theta found by optimize.minimize: {:.3f}'.format(cost))
print('Expected cost (approx): 0.203\n')

print('theta:')
print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*theta))
print('Expected theta (approx):\n\t[-25.161, 0.206, 0.201]')

# Plot Boundary
utils.plotDecisionBoundary(plot_data, theta, X.to_numpy(), y.to_numpy())
plt.savefig('./Figure/decision_boundary.png')

## ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of
#  our model.
#
#  Your task is to complete the code in predict.m

#  Predict probability for a student with score 45 on exam 1
#  and score 85 on exam 2
print('\n\n============== Part 4: Predict and Accuracies ==============')
#  Predict probability for a student with score 45 on exam 1
#  and score 85 on exam 2
prob = sigmoid(np.dot([1, 45, 85], theta))
print('For a student with scores 45 and 85,'
      'we predict an admission probability of {:.3f}'.format(prob))
print('Expected value: 0.775 +/- 0.002')

# Compute accuracy on our training set
p = predict(theta, X)
print('Train Accuracy: {:.2f} %'.format(np.mean(p == y) * 100))
print('Expected accuracy (approx): 89.00 %')
grader[4] = predict
# send the added functions to coursera grader for getting a grade on this part
ex2_reg(grader)
grader.grade()
