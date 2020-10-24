"""This is the snippet of code of neural network without using sklearn library, wirtten by Phan Tien Dung the University of Tuebingen"""

import numpy as np
import struct

class neuralnetwork:

    """neural network in machine learning"""
    def __init__(self,
                 n_hidden = 30,
                 l2 = 0,
                 epochs = 100,
                 eta = 0.001,
                 shuffle = True,
                 minibatch = 100,
                 seed = None):

        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.eta = eta
        self.epochs = epochs
        self.shuffle = shuffle
        self.minibatch = minibatch

    def _onehot(self, y, n_class):

        """one hot the output"""
        onehot = np.zeros((n_class, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):

            onehot[val, idx] = 1

        return onehot.T

    def _sigmoid(self, z):

        """sigmoid function"""
        return 1/(1 + np.exp(-np.clip(z, -250, 250)))

    def _forward(self, X):

        """forward computation"""
        hidden_in = np.dot(X, self.hidden_w) + self.hidden_b
        hidden_out = self._sigmoid(hidden_in)
        output_in = np.dot(hidden_out, self.output_w) + self.output_b
        output_out = self._sigmoid(output_in)

        return hidden_in, hidden_out, output_in, output_out

    def _compute_cost(self, y_enc, output):

        """computation cost function"""
        l2_term = (self.l2 * (np.sum(self.hidden_w**2)+np.sum(self.output_w**2)))
        term1 = -y_enc * np.log(output)
        term2 = (1-y_enc) * np.log(1 - output)
        cost = np.sum((term1 - term2)) + l2_term

        return cost

    def predict(self, X):

        """predict class label"""
        hidden_in, hidden_out, output_in, output_out = self._forward(X)
        y_pred = np.argmax(output_out)

        return y_pred

    def fit(self, X_train, y_train, X_valid, y_valid):

        """fitting model"""
        n_output = np.unique(y_train).shape[0]
        n_feature = X_train.shape[1]
        self.hidden_w = self.random.normal(0, 1, (n_feature, self.n_hidden))
        self.hidden_b = np.zeros(self.n_hidden)
        self.output_w = self.random.normal(0, 1, (self.n_hidden, n_output))
        self.output_b = np.zeros(n_output)
        epoch_strlen = len(str(self.epochs))
        self.eval_ = {

            'cost': [],
            'train_acc': [],
            'valid_acc': []

        }
        y_train_enc = self._onehot(y_train, n_output)

        for _ in range(self.epochs):

            indices = np.arange(X_train.shape[0])

            if self.shuffle:

                self.random.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - self.minibatch):

                batch_idx = indices[start_idx : (start_idx + self.minibatch)]
                hidden_in, hidden_out, output_in, output_out = self._forward(X_train[batch_idx])
                sigma_out = output_out - y_train_enc[batch_idx]
                sigmoid_derivative_hidden = hidden_out * (1- hidden_out)
                sigmoid_hidden = (np.dot(sigma_out, self.output_w.T)*sigmoid_derivative_hidden)
                grad_hidden_w = np.dot(X_train[batch_idx].T, sigmoid_hidden)
                grad_hidden_b = np.sum(sigmoid_hidden, axis = 0)
                grad_output_w = np.dot(hidden_out.T, sigma_out)
                grad_output_b = np.sum(sigma_out, axis = 0)
                delta_hidden_w = (grad_hidden_w + self.l2 * self.hidden_w)
                delta_hidden_b = grad_hidden_b
                delta_output_w = (grad_output_w + self.l2 * self.output_w)
                delta_output_b = grad_output_b
                self.hidden_w -= self.eta * delta_hidden_w
                self.hidden_b -= self.eta * delta_hidden_b
                self.output_w -= self.eta * delta_output_w
                self.output_b -= self.eta * delta_output_b

            hidden_in, hidden_out, output_in, output_out = self._forward(X_train)
            cost = self._compute_cost(y_train_enc, output_out)
            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)
            train_acc = ((y_train_pred == y_train).sum())/X_train.shape[0]
            valid_acc = ((y_valid == y_valid_pred).sum())/X_valid.shape[0]

            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)
            print(f'complete {epoch_strlen, _ +  1, self.epochs}')

        return self

def load_data(name):

    """loading train and test data"""
    with open(name + '-labels.idx1-ubyte', 'rb') as label_path:

        magic, n = struct.unpack('>II', label_path.read(8))
        label = np.fromfile(label_path, dtype=np.uint8)

    with open(name + '-images.idx3-ubyte', 'rb') as image_path:

        magic, num, row, col = struct.unpack('>IIII', image_path.read(16))
        image = np.fromfile(image_path, dtype = np.uint8).reshape(len(label), 784)
        image = ((image/255) - 0.5) * 2

    return image, label

if __name__ == '__main__':

    """python"""
    X_train, y_train = load_data('train')
    X_test, y_test = load_data('t10k')
    nn = neuralnetwork(n_hidden = 30,
                       l2 = 0.01,
                       epochs = 200,
                       eta = 0.0005,
                       minibatch = 100,
                       shuffle = True,
                       seed = 1)

    nn.fit(X_train, y_train, X_test, y_test)
    print(nn.eval_)


