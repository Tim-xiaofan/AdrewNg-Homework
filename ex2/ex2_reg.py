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
from costFunctionReg import costFunctionReg
from sigmoid import sigmoid
from predict import predict
import os

def ex2_reg(grader):
    # Load Data
    # The first two columns contains the X values and the third column
    # contains the label (y).
    data = np.loadtxt(os.path.join('Data', 'ex2data2.txt'), delimiter=',')
    X = data[:, :2]
    y = data[:, 2]

    ## ==================== Part 1: Plotting ====================
    # We start the exercise by first plotting the data to understand the
    # the problem we are working with.
    print('\n\n==================== Part 1: Plotting ====================')
    plot_data(X, y, ['Test 1 Score', 'Test 2 Score'], ['Accepted', 'Rejected'])
    plt.savefig('./Figure/dataset2.png')

    ## ============ Part 2: Compute Cost and Gradient ============
    #  In this part of the exercise, you will implement the cost and gradient
    #  for logistic regression. You neeed to complete the code in
    #  costFunction.m
    print('\n\n============ Part 2: Compute Cost and Gradient ============')
    #  Setup the data matrix appropriately, and add ones for the intercept term
    # Note that mapFeature also adds a column of ones for us, so the intercept
    # term is handled
    X = utils.mapFeature(X[:, 0], X[:, 1])
    # print('X:\n', X[:5, :])

    # Initialize fitting parameters
    initial_theta = np.zeros(X.shape[1])

    # Set regularization parameter lambda to 1
    # DO NOT use `lambda` as a variable name in python
    # because it is a python keyword
    lambda_ = 1

    # Compute and display initial cost and gradient for regularized logistic
    # regression
    cost, grad = costFunctionReg(initial_theta, X, y, lambda_)

    print('Cost at initial theta (zeros): {:.3f}'.format(cost))
    print('Expected cost (approx)       : 0.693\n')

    print('Gradient at initial theta (zeros) - first five values only:')
    print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))
    print('Expected gradients (approx) - first five values only:')
    print('\t[0.0085, 0.0188, 0.0001, 0.0503, 0.0115]\n')

    # Compute and display cost and gradient
    # with all-ones theta and lambda = 10
    test_theta = np.ones(X.shape[1])
    cost, grad = costFunctionReg(test_theta, X, y, 10)

    print('------------------------------\n')
    print('Cost at test theta    : {:.2f}'.format(cost))
    print('Expected cost (approx): 3.16\n')

    print('Gradient at test theta - first five values only:')
    print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))
    print('Expected gradients (approx) - first five values only:')
    print('\t[0.3460, 0.1614, 0.1948, 0.2269, 0.0922]')

    grader[5] = costFunctionReg
    grader[6] = costFunctionReg
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
    lambdas = [0, 1, 100]
    for i in range(3):
        theta = np.zeros(X.shape[1])
        res = optimize.minimize(fun=costFunctionReg,
                                x0=theta,
                                args=(X, y, lambdas[i]),
                                jac=True,
                                method='TNC',
                                options=options)

        # the fun property of `OptimizeResult` object returns
        # the value of costFunction at optimized theta
        cost = res.fun

        # the optimized theta is in the x property
        theta = res.x

        # Print theta to screen
        print('Cost at theta found by optimize.minimize:\n', cost)
        # print('Expected cost (approx): 0.203\n')

        # print('theta:')
        # print('\n\t', theta)
        # print('Expected theta (approx):\n\t[-25.161, 0.206, 0.201]')

        # Plot Boundary
        utils.plotDecisionBoundary(plot_data, theta, X, y,
                                   ['Test 1 Score', 'Test 2 Score'],
                                   ['Accepted', 'Rejected'], 'lambda={}'.format(lambdas[i]))
        plt.savefig('./Figure/decision_boundary2{}.png'.format(i))

        print('\n\n============= Predict if lambda={}  ============='.format(lambdas[i]))
        # Compute accuracy on our training set
        p = predict(theta, X)
        print('Train Accuracy: %.1f %%' % (np.mean(p == y) * 100))
        print('Expected accuracy (with lambda = 1): 83.1 % (approx)\n')
