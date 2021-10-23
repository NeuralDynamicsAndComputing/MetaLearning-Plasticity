import torch
import warnings
import argparse

import numpy as np

from torch import nn, optim
from torchviz import make_dot
from torch.nn.utils import _stateless
from torch.utils.data import DataLoader, RandomSampler, Dataset
# from kymatio.torch import Scattering2D

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

        # -- network params
        self.fc1 = nn.Linear(self.in_dim, 512)
        self.fc2 = nn.Linear(512, 264)
        self.fc3 = nn.Linear(264, 128)
        self.fc4 = nn.Linear(128, 964)

        self.relu = nn.ReLU()

        # -- learning params
        self.alpha = nn.Parameter(torch.randn(1))
        self.beta = nn.Parameter(torch.randn(1))

    def forward(self, y0):

        y1 = self.relu(self.fc1(y0))
        y2 = self.relu(self.fc2(y1))
        y3 = self.relu(self.fc3(y2))

        return (y0, y1, y2, y3), self.fc4(y3)


class Train:
    def __init__(self, train_dataloader, args):

        # -- model params
        self.model = MyModel()
        # self.scat = Scattering2D(J=3, L=8, shape=(28, 28), max_order=2)
        self.softmax = nn.Softmax(dim=1)
        self.n_layers = 4  # fixme

        # -- training params
        self.epochs = args.epochs

        # -- data params
        self.TrainDataloader = train_dataloader
        self.N = args.N

        # -- optimization params
        self.lr_innr = args.lr_innr  # fixme
        self.lr_meta = args.lr_meta  # fixme
        self.loss_func = nn.CrossEntropyLoss()
        self.optim_meta = optim.Adam(self.model.parameters(), lr=self.lr_meta)
        # self.optim_innr = MyOptimizer(self.model.parameters(), lr=self.lr_innr)  # todo: pass network params only

    def __call__(self):
        """
            Model training.
        """
        self.model.train()
        for episode, data in enumerate(self.TrainDataloader):

            train_loss = 0

            # -- training data
            img_trn, lbl_trn, img_tst, lbl_tst = process_data(data)
            params = dict(self.model.named_parameters())

            """ inner update """
            for image, label in zip(img_trn, lbl_trn):
                params = {k: v.clone() for k, v in params.items()}

                # -- predict
                image = image.reshape(1, -1)
                _, logits = _stateless.functional_call(self.model, params, image)

                if False:
                    make_dot(logits, params=dict(list(self.model.named_parameters()))).render('model_torchviz', format='png')
                    quit()

                # -- compute loss
                loss_inner = self.loss_func(logits, label)

                # -- update network params
                loss_inner.backward(create_graph=True, inputs=params.values())
                params = MyOptimizer(params, self.model.alpha)

            """ meta update """
            # -- predict
            _, logits = _stateless.functional_call(self.model, params, img_tst.reshape(25, -1))
            if False:
                make_dot(logits, params=dict(list(self.model.named_parameters()))).render('model_torchviz', format='png')
                quit()

            # -- compute loss
            loss_meta = self.loss_func(logits, lbl_tst.reshape(-1))

            # -- update params
            self.optim_meta.zero_grad()
            loss_meta.backward()
            train_loss += loss_meta.item()
            self.optim_meta.step()

            # -- log
            print('Train episode: {}\tLoss: {:.6f}'.format(episode, train_loss / (self.N * 5)))


def parse_args():
    desc = "Numpy implementation of mnist label predictor."
    parser = argparse.ArgumentParser(description=desc)

    # -- training params
    parser.add_argument('--epochs', type=int, default=3000, help='The number of epochs to run.')

    parser.add_argument('--N', type=int, default=400, help='Number of training data.')

    # -- meta-training params
    parser.add_argument('--steps', type=int, default=5, help='.')  # fixme: add definition
    parser.add_argument('--tasks', type=int, default=5, help='.')  # fixme: add definition
    parser.add_argument('--lr_innr', type=float, default=1e-3, help='.')
    parser.add_argument('--lr_meta', type=float, default=1e-3, help='.')

    return parser.parse_args()


def main():
    args = parse_args()

    # -- load data
    M = 100
    dataset = OmniglotDataset(steps=args.steps, N=args.N)
    sampler = RandomSampler(data_source=dataset, replacement=True, num_samples=M)
    dataloader = DataLoader(dataset=dataset, sampler=sampler, batch_size=args.tasks, drop_last=True)

    # -- train model
    my_train = Train(dataloader, args)
    my_train()


if __name__ == '__main__':
    main()
