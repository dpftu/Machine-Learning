"""Logistic Regression implemented manually"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

class logisticregression:

    def __init__(self, lr = 0.01, epoch = 10, random_state = 5, method = 'sgd'):

        self.lr = lr
        self.epoch = epoch
        self.random_state = random_state
        self.method = method

    def fit(self, X, y):

        if self.method == 'sgd':

            np.random.seed(self.random_state)
            self.coef_ = np.random.normal(0, 1, X.shape[1]+1)
            self.total_cost_ = []

            for _ in range(self.epoch):

                cost = []

                for xi, target in zip(X, y):

                    netinput = self.netinput(xi)
                    output = self.activation(netinput)
                    error = target - output
                    self.coef_[1:] += self.lr * xi * error
                    self.coef_[0] += self.lr * error
                    cost_of_obs = -target*np.log(output) - ((1-target)*np.log(1-output))
                    cost.append(cost_of_obs)

                cost_of_batch = sum(cost)/len(y)
                self.total_cost_.append(cost_of_batch)

            for i, j in enumerate(self.total_cost_):

                print(f'epoch {i+1}, loss {j}')

        else:

            np.random.seed(self.random_state)
            self.coef_ = np.random.normal(0, 1, X.shape[1]+1)
            self.total_cost_ = []

            for _ in range(self.epoch):

                netinput = self.netinput(X)
                output = self.activation(netinput)
                error = y - output
                self.coef_[1:] += self.lr * np.dot(X.T, error)
                self.coef_[0] +=  self.lr * error.sum()
                cost_of_batch = -y.dot(np.log(output)) - ((1-y).dot(np.log(1-output)))
                self.total_cost_.append(cost_of_batch)

            for i, j in enumerate(self.total_cost_):

                print(f'epoch {i+1}, loss {j}')


        return self

    def netinput(self, X):

        return np.dot(X, self.coef_[1:]) + self.coef_[0]

    def activation(self, z):

        return 1/(1+np.exp(-np.clip(z, -250, 250)))

    def prediction(self, X):

        return np.where(self.activation(self.netinput(X)) >= 1/2, 1, 0)

if __name__ == '__main__':

    """machine learning in python"""

    iris = load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    print(f'group: {np.unique(y)}')

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=5
    )

    X_train_subset = X_train[(y_train == 0) | (y_train == 1)]
    y_train_subset = y_train[(y_train == 0) | (y_train == 1)]
    X_test_subset = X_test[(y_test == 0) | (y_test == 1)]
    y_test_subset = y_test[(y_test == 0) | (y_test == 1)]
    logistic = logisticregression(lr = 0.01, epoch = 100, random_state = 5, method = 'gd')
    logistic.fit(X_train_subset, y_train_subset)
    print(f'test acc: {(logistic.prediction(X_test_subset) == y_test_subset).sum()/len(y_test_subset)}')


