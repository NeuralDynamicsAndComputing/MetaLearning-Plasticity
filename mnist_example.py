import numpy as np
import torch
from keras.datasets import mnist
import argparse


def load_data(n_train):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = np.reshape(x_train[:n_train, :, :], (n_train, 784)).T
    y_train = np.reshape(y_train[:n_train], (n_train, 1)).T

    return x_train, y_train, x_test, y_test


class Linear:
    def __init__(self, input_size, output_size):
        w = 1. / np.sqrt(input_size)
        self.weight = np.random.uniform(-w, w, (output_size, input_size))

    def __call__(self, x):
        return np.matmul(self.weight, x)


class Model:
    def __init__(self):
        self.h_1 = Linear(784, 1500)
        self.h_2 = Linear(1500, 1500)
        self.h_3 = Linear(1500, 1500)
        self.h_4 = Linear(1500, 1)

    @property
    def get_layers(self):
        return {1: self.h_1, 2: self.h_2, 3: self.h_3, 4: self.h_4}

    @staticmethod
    def relu(x):
        return np.maximum(np.zeros(x.shape), x)

    def __call__(self, y0):
        y1 = self.relu(self.h_1(y0))
        y2 = self.relu(self.h_2(y1))
        y3 = self.relu(self.h_3(y2))
        y4 = self.relu(self.h_4(y3))

        return y0, y1, y2, y3, y4


class Train:
    def __init__(self, x_train, y_train, args):
        self.model = Model()
        self.epochs = args.epochs
        self.n_layers = len(self.model.get_layers)
        self.eta = args.eta
        self.batch_size = args.batch_size
        self.N = len(x_train.T)
        self.batch_n = -(-self.N//args.batch_size)

        self.X_train = x_train
        self.y_train = y_train

    def del_w(self, y_lm1, e_l):
        """
            Weight update rule.
        :param y_lm1: inputs to the layer.
        :param e_l: error vector.
        :return: weight update
        """
        return - self.eta * np.matmul(e_l, y_lm1.T)

    def train_epoch(self, epoch):
        """
            Single epoch training.
        :param epoch: current epoch number.
        """
        train_loss = 0
        for idx in range(self.batch_n):

            # -- training data
            y0 = self.X_train[:, idx * self.batch_size:(idx + 1) * self.batch_size]/256
            y_target = self.y_train[:, idx * self.batch_size:(idx + 1) * self.batch_size]

            # -- predict
            y = self.model(y0)

            # -- compute error
            e = [y[-1] - y_target]
            for i in range(4, 1, -1):
                e.insert(0, np.matmul(self.model.get_layers[i].weight.T, e[0]) * np.heaviside(y[i-1], 0.0))

            # -- weight update
            for i, key in enumerate(self.model.get_layers.keys()):
                self.model.get_layers[key].weight = self.model.get_layers[key].weight + self.del_w(y[i], e[i])

            # -- compute loss
            train_loss += 0.5 * np.matmul(e[-1], e[-1].T).item()

        # -- log
        print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, train_loss / self.N))

    def __call__(self):
        """
            Model training.
        """

        for epoch in range(1, self.epochs+1):
            self.train_epoch(epoch)


def parse_args():
    desc = "Numpy implementation of mnist label predictor."
    parser = argparse.ArgumentParser(description=desc)

    # -- training params
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs to run.')
    parser.add_argument('--batch_size', type=int, default=10, help='The size of each batch.')
    parser.add_argument('--N', type=int, default=200, help='Number of training data.')
    parser.add_argument('--eta', type=float, default=1e-3, help='Learning rate.')

    return parser.parse_args()


def main():
    args = parse_args()

    x_train, y_train, _, _ = load_data(args.N)
    my_train = Train(x_train, y_train, args)

    my_train()


if __name__ == '__main__':
    main()
