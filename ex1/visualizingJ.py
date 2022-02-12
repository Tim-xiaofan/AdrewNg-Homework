from computeCost import computeCost
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np


def visualizingJ(X, y, theta):
    # Grid over which we will calculate J
    theta0 = np.linspace(-10, 10, 100)
    theta1 = np.linspace(-1, 4, 100)

    # initialize J_vals to a matrix of 0's
    J = np.zeros((len(theta0), len(theta1)))

    # Fill out J_vals
    for i in range(len(theta0)):
        for j in range(len(theta1)):
            t = np.array([theta0[i], theta1[j]])
            J[i, j] = computeCost(X, y, t)

    # Because of the way meshgrids work in the surf command, we need to
    # transpose J before calling surf, or else the axes will be flipped(翻转)
    J = J.T

    # create 2-D meshgrid
    theta0, theta1 = np.meshgrid(theta0, theta1)

    # Surface plot 曲面图
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(theta0, theta1, J, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
    ax.set_xlabel(r'${\Theta}$0')
    ax.set_ylabel(r'${\Theta}$1')
    ax.set_zlabel('J')
    ax.set_title('Surface')
    plt.show()

    # Contour plot 绘制登高线图
    fig, ax = plt.subplots()
    # Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
    # 将 J 绘制为 15 个在 0.01 和 100 之间对数间隔的等高线
    CS = ax.contour(theta0, theta1, J, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_title('Contour')
    ax.set_xlabel(r'${\Theta}$0')
    ax.set_ylabel(r'${\Theta}$1')
    plt.plot(theta[0], theta[1], 'rx', ms=10, lw=2)
    plt.legend(['Minimum'])
    plt.show()
