from matplotlib import pyplot as plt
import pandas as pd


def plot_data(X, y, labels=None, legengs=None, title=None):
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    data = pd.concat([X, y], axis=1)
    positive = data[data.iloc[:, -1].isin([1])]
    negative = data[data.iloc[:, -1].isin([0])]

    fig, ax = plt.subplots()
    if legengs is None:
        ax.scatter(positive.iloc[:, 0], positive.iloc[:, 1], s=50, c='b', marker='o', label='Admitted')
        ax.scatter(negative.iloc[:, 0], negative.iloc[:, 1], s=50, c='r', marker='x', label='Not Admitted')
    else:
        ax.scatter(positive.iloc[:, 0], positive.iloc[:, 1], s=50, c='b', marker='o', label=legengs[0])
        ax.scatter(negative.iloc[:, 0], negative.iloc[:, 1], s=50, c='r', marker='x', label=legengs[1])
    ax.legend()
    if labels is None:
        ax.set_xlabel('Exam 1 Score')
        ax.set_ylabel('Exam 2 Score')
    else:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
    if title is not None:
        ax.set_title(title)
    plt.show()
