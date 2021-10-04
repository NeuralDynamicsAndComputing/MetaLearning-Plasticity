import torch
import warnings
import argparse

import numpy as np

from torch import nn
import torch.optim as optim
from Dataset import OmniglotDataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from kymatio.torch import Scattering2D

warnings.simplefilter(action='ignore', category=UserWarning)

np.random.seed(0)


def Myload_data(data, tasks=5, steps=5, iid=True):

    img_trn, lbl_trn, img_tst, lbl_tst = data

    img_tst = torch.reshape(img_tst, (tasks * 5,  28, 28))
    lbl_tst = torch.reshape(lbl_tst, (tasks * 5, 1))

    img_trn = torch.reshape(img_trn, (tasks * steps, 28, 28))
    lbl_trn = torch.reshape(lbl_trn, (tasks * steps, 1))

    if iid:
        perm = np.random.choice(range(tasks * steps), tasks * steps, False)

        img_trn = img_trn[perm]
        lbl_trn = lbl_trn[perm]

    return img_trn, lbl_trn, img_tst, lbl_tst


class Model:  # todo: merge with MyModel

    @property
    def get_layers(self):  # fixme: might need
        return {1: self.h_1, 2: self.h_2, 3: self.h_3, 4: self.h_4}

    @property
    def feedback_matrix(self):  # fixme: keep

        # todo: define B as network parameter
        feed_mat = {}
        for i in range(1, len(self.get_layers)):  # todo: find a better way to get an iterator over network params
            feed_mat[i] = self.get_layers[i+1].weight.T  # todo: may need to change init of B.

        return feed_mat

    def __call__(self, y0):
        y1 = self.relu(self.h_1(y0))
        y2 = self.relu(self.h_2(y1))
        y3 = self.relu(self.h_3(y2))
        y4 = self.relu(self.h_4(y3))

        return y0, y1, y2, y3, y4  # fixme: might need


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # -- dim
        self.in_dim = 1953

        # -- network parameters
        self.fc1 = nn.Linear(self.in_dim, 512)  # , bias=False)
        self.fc2 = nn.Linear(512, 264)  # , bias=False)
        self.fc3 = nn.Linear(264, 128)  # , bias=False)
        self.fc4 = nn.Linear(128, 964)  # , bias=False)

        self.relu = nn.ReLU()

    def forward(self, y0):

        y1 = self.relu(self.fc1(y0))
        y2 = self.fc2(y1)
        y3 = self.fc3(y2)

        return self.fc4(y3)


class Train:
    def __init__(self, trainset, args):

        # -- model params
        self.model = MyModel()
        self.scat = Scattering2D(J=3, L=8, shape=(28, 28), max_order=2)

        # self.B = self.model.feedback_matrix  # todo: redefine in MyModel
        # self.n_layers = len(self.model.get_layers)  # fixme

        # -- training params
        self.eta = args.eta
        self.epochs = args.epochs
        self.batch_size = args.batch_size  # todo: remove
        self.batch_n = -(-args.N//args.batch_size)  # todo: remove

        # -- data params
        self.TrainDataset = trainset

        # -- optimization params
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-2)  # todo: switch this w/ costume update rule

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

    def train_epoch(self, epoch):
        """
            Single epoch training.
        :param epoch: current epoch number.
        """
        self.model.train()
        train_loss = 0

        for batch_idx, data in enumerate(self.TrainDataset):

            # -- training data # todo: swap w/ Omniglot dataloader and call to 'Myload_data'
            img_trn, lbl_trn, img_tst, lbl_tst = Myload_data(data)

            for image, label in zip(img_trn, lbl_trn):

                # if batch_idx < 10:  # fixme
                self.optimizer.zero_grad()

                # -- predict
                y4 = self.model(self.scat(image).reshape(1, -1))

                # -- compute loss
                loss = self.loss_func(y4, label.to(torch.float32))

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

    # -- meta-training params
    parser.add_argument('--steps', type=int, default=5, help='')  # fixme: add definition
    parser.add_argument('--tasks', type=int, default=5, help='')  # fixme: add definition

    return parser.parse_args()


def main():
    args = parse_args()

    # -- load data
    train_dataset = DataLoader(dataset=OmniglotDataset(args.steps), batch_size=args.tasks, shuffle=True, drop_last=True)

    # -- train model
    my_train = Train(train_dataset, args)
    my_train()


if __name__ == '__main__':
    main()
