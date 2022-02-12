import seaborn as sns
from matplotlib import pyplot as plt

sns.set(context="notebook", style="whitegrid", palette="dark")


def plotData(x, y):
    """
    Plots the data points x and y into a new figure. Plots the data
    points and gives the figure axes labels of population and profit.

    Parameters
    ----------
    x : array_like
        Data point values for x-axis.

    y : array_like
        Data point values for y-axis. Note x and y should have the same size.

    Instructions
    ------------
    Plot the training data into a figure using the "figure" and "plot"
    functions. Set the axes labels using the "xlabel" and "ylabel" functions.
    Assume the population and revenue data have been passed in as the x
    and y arguments of this function.

    Hint(提示)
    ----
    You can use the 'ro' option with plot to have the markers
    appear as red circles. Furthermore, you can make the markers larger by
    using plot(..., 'ro', ms=10), where `ms` refers to marker size. You
    can also set the marker edge color using the `mec` property.
    """
    # sns.lmplot(x='population', y='profit', data=df,
    #            height=6, fit_reg=False, markers='x',
    #            scatter_kws={'color': 'red'})
    # plt.show()
    fig, ax = plt.subplots()
    ax.plot(x, y, 'rx', ms=10, label='Training Data')
    plt.legend()
    ax.set_xlabel('population')
    ax.set_ylabel('profit')
    ax.set_title('Training Data')
