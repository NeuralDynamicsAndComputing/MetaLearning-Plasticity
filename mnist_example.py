import torch
import warnings
import argparse

import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)

from keras.datasets import mnist
from torch import nn
import torch.optim as optim
from Dataset import OmniglotDataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

np.random.seed(0)


def load_data(n_train):  # todo: swap w/ 'Myload_data'
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = np.reshape(x_train[:n_train, :, :], (n_train, 784)).T
    y_train = np.reshape(y_train[:n_train], (n_train, 1)).T

    return x_train, y_train, x_test, y_test


def Myload_data(data, tasks=5, steps=5, iid=True):

    img_trn, lbl_trn, img_tst, lbl_tst = data

    img_tst = torch.reshape(img_tst, (tasks * 5,  84, 84))
    lbl_tst = torch.reshape(lbl_tst, (1, tasks * 5))

    img_trn = torch.reshape(img_trn, (tasks * steps, 84, 84))
    lbl_trn = torch.reshape(lbl_trn, (tasks * steps, 1))

    if iid:
        perm = np.random.choice(range(tasks * steps), tasks * steps, False)

        img_trn = img_trn[perm]
        lbl_trn = lbl_trn[perm]

    return img_trn, lbl_trn, img_tst, lbl_tst


class Linear:  # todo: remove
    def __init__(self, input_size, output_size):
        w = 1. / np.sqrt(input_size)
        self.weight = np.random.uniform(-w, w, (output_size, input_size))

    def __call__(self, x):
        return np.matmul(self.weight, x)


class Model:
    def __init__(self):  # todo: remove
        self.h_1 = Linear(784, 512)
        self.h_2 = Linear(512, 256)
        self.h_3 = Linear(256, 128)
        self.h_4 = Linear(128, 1)

    @property
    def get_layers(self):  # fixme: might need
        return {1: self.h_1, 2: self.h_2, 3: self.h_3, 4: self.h_4}

    @property
    def feedback_matrix(self):   # fixme: keep

        # todo: define B as network parameter
        feed_mat = {}
        for i in range(1, len(self.get_layers)):  # todo: find a better way to get an iterator over network weights
            feed_mat[i] = self.get_layers[i+1].weight.T  # todo: may need to change init of B.

        return feed_mat

    @staticmethod
    def relu(x):  # todo: remove
        return np.maximum(np.zeros(x.shape), x)

    def __call__(self, y0):
        y1 = self.relu(self.h_1(y0))
        y2 = self.relu(self.h_2(y1))
        y3 = self.relu(self.h_3(y2))
        y4 = self.relu(self.h_4(y3))

        return y0, y1, y2, y3, y4  # fixme: might need


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # -- network parameters
        self.fc1 = nn.Linear(784, 512, bias=False)  # todo: 784 -> 84*84
        self.fc2 = nn.Linear(512, 264, bias=False)
        self.fc3 = nn.Linear(264, 128, bias=False)
        self.fc4 = nn.Linear(128, 1, bias=False)  # todo: 1 -> 5 or 963

        self.relu = nn.ReLU()

    def forward(self, y0):

        y1 = self.relu(self.fc1(y0.squeeze(1).reshape(-1, 784)))  # todo: 784 -> 84*84
        y2 = self.relu(self.fc2(y1))
        y3 = self.relu(self.fc3(y2))

        return self.relu(self.fc4(y3))


class Train:
    def __init__(self, x_train, y_train, args):

        # -- model params
        self.model = MyModel()  # todo: remove (use self.model = MyModel())

        # self.B = self.model.feedback_matrix  # todo: redefine in MyModel
        # self.n_layers = len(self.model.get_layers)  # fixme

        # -- training params
        self.eta = args.eta
        self.X_train = x_train  # todo: define dataloader here!
        self.y_train = y_train  # todo: remove
        self.N = len(x_train.T)  # fixme
        self.epochs = args.epochs
        self.batch_size = args.batch_size  # todo: remove
        self.batch_n = -(-self.N//args.batch_size)  # todo: remove

        # todo: swap w/ Omniglot dataloader
        mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        self.TrainDataset = DataLoader(mnist_trainset, batch_size=self.batch_size, shuffle=False)

        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3)  # todo: switch this w/ costume update rule

    def feedback_update(self, y):  # fixme: use pytorch functions to update B matrix?
        """
            updates feedback matrix B.
        :param y: input, activations, and prediction
        :return:
        """
        for i in range(1, self.n_layers):
            self.B[i] -= self.eta * np.matmul(self.e[i], y[i].T).T

    def weight_update(self, y, y_target):
        """
            Weight update rule.
        :param y: input, activations, and prediction
        :param y_target: target label

        todo: plan:
        1) get simple pytorch code working w/o this
        2) compute these and try to update W
        3) try changing update rule to a learnable one modify to work w/ pytorch
        """

        # -- compute error
        self.e = [y[-1] - y_target]
        for i in range(self.n_layers, 1, -1):
            self.e.insert(0, np.matmul(self.B[i-1], self.e[0]) * np.heaviside(y[i-1], 0.0))

        # -- weight update
        for i, key in enumerate(self.model.get_layers.keys()):
            self.model.get_layers[key].weight = self.model.get_layers[key].weight - \
                                                self.eta * np.matmul(self.e[i], y[i].T)

    def train_epoch_(self, epoch):  # todo: remove
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

            # -- weight update
            self.weight_update(y, y_target)

            # -- feedback update
            self.feedback_update(y)

            # -- compute loss
            train_loss += 0.5 * np.matmul(self.e[-1], self.e[-1].T).item()

        # -- log
        print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, train_loss / self.N))

    def train_epoch(self, epoch):
        """
            Single epoch training.
        :param epoch: current epoch number.
        """
        self.model.train()
        train_loss = 0

        for batch_idx, data in enumerate(self.TrainDataset):

            # -- training data # todo: swap w/ Omniglot dataloader and call to 'Myload_data'
            img, label = data

            if batch_idx < 10:  # fixme
                self.optimizer.zero_grad()

                # -- predict
                y4 = self.model(img)

                # -- compute loss
                e = y4 - label.unsqueeze(1)
                loss = 0.5 * e.T @ e

                # -- weight update todo: 1) compute W updates w/ error and feedback, 2) switch to a costume update rule
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()

                # -- feedback update
                # todo: 1) define feedback and its update rule 2) meta learn feedback

        # -- log
        print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, train_loss / 200))

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
