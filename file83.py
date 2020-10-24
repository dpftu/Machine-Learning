

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt



class perceptron:

    """perceptron algorithm in python"""
    def __init__(self, lr = 0.01, epoch = 20, random_state = 5):

        self.lr = lr
        self.epoch = epoch
        self.random_state = random_state

    def fit(self, X, y):

        rgen = np.random.RandomState(self.random_state)
        self.coef_ = np.random.normal(0, 1, X.shape[1] + 1)
        self.error_ = []

        for _ in range(self.epoch):

            error = 0

            for xi, yi in zip(X, y):

                update = (yi - self.predict(xi))
                self.coef_[1: ] += self.lr*update*xi
                self.coef_[0] += self.lr*update
                error += int(update != 0)

            self.error_.append(error)


        return self

    def netinput(self, X):

        return np.dot(X, self.coef_[1 : ]) + self.coef_[0]

    def predict(self, X):

        return np.where(self.netinput(X) >= 0, 1, -1)


def plot_decision_region(X, y, classifier, resolution = 0.02):


    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 0].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.3, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):

        plt.scatter(

            x = X[y == cl, 0],
            y = X[y == cl, 1],
            alpha = 0.8,
            c = colors[idx],
            marker = markers[idx],
            label = cl,
            edgecolors = 'black'

        )

if __name__ == '__main__':


    """python programming"""

    df = pd.read_csv('iris.csv', header=None)
    X = df.iloc[0: 100, [0, 2]].values
    y = df.iloc[0: 100, 4].values
    y = np.where(y == 'Iris-setosa', 1, -1)
    perceptron = perceptron(lr=0.01, epoch=20, random_state=5)
    perceptron.fit(X, y)
    plot_decision_region(X, y, classifier=perceptron)
    plt.xlabel('Sepal Length')
    plt.ylabel('Petal Length')
    plt.legend(loc='upper right')
    plt.show()





























