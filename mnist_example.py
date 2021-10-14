import torch
import warnings
import argparse

import numpy as np

from torch import nn, optim
from torchviz import make_dot
from kymatio.torch import Scattering2D
from torch.utils.data import DataLoader

from Optim_rule import MyOptimizer
from Dataset import OmniglotDataset, process_data

warnings.simplefilter(action='ignore', category=UserWarning)

np.random.seed(0)
torch.manual_seed(0)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # -- dim
        self.in_dim = 784

        # -- network parameters
        self.fc1 = nn.Linear(self.in_dim, 512)  # , bias=False)
        self.fc2 = nn.Linear(512, 264)  # , bias=False)
        self.fc3 = nn.Linear(264, 128)  # , bias=False)
        self.fc4 = nn.Linear(128, 964)  # , bias=False)

        self.relu = nn.ReLU()

        # -- set feedback and feedforward param lists
        self.set_param_lists()

    def set_param_lists(self):

        self.feed_fwd_params_list = nn.ParameterList([
            self.fc1.weight,
            self.fc1.bias,
            self.fc2.weight,
            self.fc2.bias,
            self.fc3.weight,
            self.fc3.bias,
            self.fc4.weight,
            self.fc4.bias
        ])

        self.feed_bck_params_list = nn.ParameterList([
            nn.Parameter(self.fc2.weight.detach().clone().T),
            nn.Parameter(self.fc3.weight.detach().clone().T),
            nn.Parameter(self.fc4.weight.detach().clone().T)
        ])

    def forward(self, y0):

        y1 = self.relu(self.fc1(y0))
        y2 = self.relu(self.fc2(y1))
        y3 = self.relu(self.fc3(y2))

        return (y1, y2, y3), self.fc4(y3)


class Train:
    def __init__(self, trainset, args):

        # -- model params
        self.model = MyModel()
        self.scat = Scattering2D(J=3, L=8, shape=(28, 28), max_order=2)
        self.softmax = nn.Softmax(dim=1)
        self.n_layers = 4

        # self.B = self.model.feedback_matrix  # todo: redefine in MyModel
        # self.n_layers = len(self.model.get_layers)  # fixme

        # -- training params
        self.epochs = args.epochs

        # -- data params
        self.TrainDataset = trainset

        # -- optimization params
        self.lr_innr = args.lr_innr
        self.lr_meta = args.lr_meta
        self.loss_func = nn.CrossEntropyLoss()
        self.optim_meta = optim.Adam(self.model.parameters(), lr=self.lr_meta)  # todo: pass only meta params
        self.optim_innr = MyOptimizer(self.model.feed_fwd_params_list, lr=self.lr_innr)  # todo: pass only weight params

    def feedback_update(self, y):
        """
            updates feedback matrix B.
        :param y: input, activations, and prediction
        :return:
        """
        for i in range(1, self.n_layers):
            # self.B[i] -= self.lr_meta * np.matmul(self.e[i], y[i].T).T
            self.B[i]  # todo: get model params

    def inner_update_(self, image, target):
        # TODO: remove
        """
            inner update rule.
        :param image: input
        :param target: target label

        todo: plan:
        2) compute these and try to update W
        3) try changing update rule to a learnable one modify to work w/ pytorch
        """

        image = image.reshape(1, -1)
        y, logits = self.model(image)
        loss = self.loss_func(logits, target)

        grad = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)

        with torch.no_grad():
            for idx, param in enumerate(self.model.parameters()):
                new_param = param - self.lr_innr * grad[idx]
                param.copy_(new_param)

        # -- compute error todo: would computations stand in the case of logit outputs?
        e = [self.softmax(logits) - target]
        for i in range(self.n_layers, 1, -1):
            e.insert(0, np.matmul(self.B[i-1], e[0]) * np.heaviside(y[i-2], 0.0))  # fixme : y[i-1] -> y[i-2]

        # # -- weight update
        # for i, key in enumerate(self.model.get_layers.keys()):
        #     self.model.get_layers[key].weight = self.model.get_layers[key].weight - \
        #                                         self.lr_innr * np.matmul(self.e[i], y[i].T)

    def inner_update(self, image, target):
        # TODO: remove
        """
            inner update rule.
        :param image: input
        :param target: target label

        todo: plan:
        2) compute these and try to update W
        3) try changing update rule to a learnable one modify to work w/ pytorch
        """

        y, logits = self.model(image.reshape(1, -1))

        loss = self.loss_func(logits, target)

        grad = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)

        with torch.no_grad():
            for idx, param in enumerate(self.model.parameters()):
                new_param = param - self.lr_innr * grad[idx]
                param.copy_(new_param)

    def train_epoch(self, epoch):
        """
            Single epoch training.
        :param epoch: current epoch number.
        """
        self.model.train()
        train_loss = 0

        for batch_idx, data in enumerate(self.TrainDataset):  # fixme: this way each X is only observed once.

            # -- training data
            img_trn, lbl_trn, img_tst, lbl_tst = process_data(data)

            """ inner update """
            for image, label in zip(img_trn, lbl_trn):
                # -- predict
                image = image.reshape(1, -1)
                y, logits = self.model(image)

                if False:
                    make_dot(logits, params=dict(list(self.model.named_parameters()))).render('model_torchviz', format='png')
                    quit()

                # -- compute loss
                loss_innr = self.loss_func(logits, label)

                # -- update params
                # todo: 1) compute W updates w/ error and feedback, 2) custom update rule
                self.optim_innr.step(loss_innr, image, y, logits, self.model.feed_bck_params_list) # todo: use that register thing (!) to call from opt func w/o passing all these info.

            """ meta update """
            # -- predict
            _, logits = self.model(img_tst.reshape(25, -1))  # self.model(self.scat(image).reshape(1, -1))

            # -- compute loss
            loss_meta = self.loss_func(logits, lbl_tst.reshape(-1))

            # -- update params
            # todo: 1) define feedback and its update rule 2) meta learn feedback
            self.optim_meta.zero_grad()
            loss_meta.backward()
            train_loss += loss_meta.item()
            self.optim_meta.step()

        # -- log
        print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, train_loss / 200))  # fixme: data size: 200 -> ??

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
    parser.add_argument('--epochs', type=int, default=3000, help='The number of epochs to run.')
    parser.add_argument('--N', type=int, default=200, help='Number of training data.')  # fixme

    # -- meta-training params
    parser.add_argument('--steps', type=int, default=5, help='.')  # fixme: add definition
    parser.add_argument('--tasks', type=int, default=5, help='.')  # fixme: add definition
    parser.add_argument('--lr_innr', type=float, default=1e-3, help='.')
    parser.add_argument('--lr_meta', type=float, default=1e-3, help='.')

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
