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

n = 84  # fixme
nxn = n * n

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # -- dim
        self.in_dim = nxn

        # -- network params
        self.fc1 = nn.Linear(self.in_dim, 512)
        self.fc2 = nn.Linear(512, 264)
        self.fc3 = nn.Linear(264, 128)
        self.fc4 = nn.Linear(128, 964)

        self.relu = nn.Softplus(beta=10)

        # -- learning params
        self.alpha = nn.Parameter(torch.rand(1) / 100)
        self.beta = nn.Parameter(torch.rand(1) / 100)

    def forward(self, y0):

        y1 = self.relu(self.fc1(y0))
        y2 = self.relu(self.fc2(y1))
        y3 = self.relu(self.fc3(y2))

        return (y0, y1, y2, y3), self.fc4(y3)


class NewModel(nn.Module):
    def __init__(self):
        super(NewModel, self).__init__()
        # -- network params
        self.cn1 = nn.Conv2d(1, 256, kernel_size=3, stride=2)
        self.cn2 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.cn3 = nn.Conv2d(256, 256, kernel_size=3, stride=2)
        self.cn4 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.cn5 = nn.Conv2d(256, 256, kernel_size=3, stride=2)
        self.cn6 = nn.Conv2d(256, 256, kernel_size=3, stride=2)

        self.fc1 = nn.Linear(2304, 1700)
        self.fc2 = nn.Linear(1700, 1200)
        self.fc3 = nn.Linear(1200, 964)

        self.relu = nn.ReLU()

        # -- learning params
        self.alpha = nn.Parameter(torch.rand(1) / 100)
        self.beta = nn.Parameter(torch.rand(1) / 100)

        # -- learnable params
        self.params = nn.ParameterList()

    def forward(self, x):

        y1 = self.relu(self.cn1(x))
        y2 = self.relu(self.cn2(y1))
        y3 = self.relu(self.cn3(y2))
        y4 = self.relu(self.cn4(y3))
        y5 = self.relu(self.cn5(y4))
        y6 = self.relu(self.cn6(y5))

        y6 = y6.view(y6 .size(0), -1)

        y7 = self.relu(self.fc1(y6))
        y8 = self.relu(self.fc2(y7))

        return (y6, y7, y8), self.fc3(y8)


class Train:
    def __init__(self, trainset, args):

        # -- model params
        path_pretrained = './data/models/omniglot_example/model_stat.pth'
        self.model = self.load_model(path_pretrained)
        # self.scat = Scattering2D(J=3, L=8, shape=(28, 28), max_order=2)
        self.softmax = nn.Softmax(dim=1)
        self.n_layers = 4  # fixme

        # -- data params
        self.TrainDataset = trainset

        # -- optimization params
        self.lr_meta = args.lr_meta
        self.loss_func = nn.CrossEntropyLoss()
        self.optim_meta = optim.Adam(self.model.params.parameters(), lr=self.lr_meta)

    def load_model(self, path_pretrained):
        """
            Loads pretrained parameters for the convolutional layers.
        """
        new_model = NewModel()
        old_model = torch.load(path_pretrained)
        for old_key in old_model:
            dict(new_model.named_parameters())[old_key].data = old_model[old_key]

        for key, val in new_model.named_parameters():
            if 'cn' in key:
                val.meta, val.adapt = False, False
            elif 'fc' in key:
                val.meta, val.adapt = True, True
            else:
                val.meta, val.adapt = True, False

            # -- learnable params
            if val.meta == True:
                new_model.params.append(val)

        return new_model

    def __call__(self):
        """
            Model training.
        """
        self.model.train()
        for episode_idx, data in enumerate(self.TrainDataset):

            train_loss = 0

            # -- training data
            img_trn, lbl_trn, img_tst, lbl_tst = process_data(data)
            params = dict(self.model.named_parameters())

            """ adaptation """
            for image, label in zip(img_trn, lbl_trn):
                params = {key: val.clone() for key, val in params.items()}
                for key in params:
                    params[key].adapt = dict(self.model.named_parameters())[key].adapt

                # -- predict
                _, logits = _stateless.functional_call(self.model, params, image.unsqueeze(0).unsqueeze(0))

                if False:
                    make_dot(logits, params=dict(list(self.model.named_parameters()))).render('model_torchviz', format='png')
                    quit()

                # -- compute loss
                loss_inner = self.loss_func(logits, label)

                # -- update network params
                loss_inner.backward(create_graph=True, inputs=params.values())
                params = MyOptimizer(params, self.model.alpha, self.model.beta)

            """ meta update """
            # -- predict
            _, logits = _stateless.functional_call(self.model, params, img_tst.unsqueeze(1))
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
            print('Train Episode: {}\tLoss: {:.6f}\tlr: {:.6f}\tdr: {:.6f}'.format(episode_idx, loss_meta.item() / 25,
                                                                                   self.model.alpha.detach().numpy()[0],
                                                                                   self.model.beta.detach().numpy()[0]))


def parse_args():
    desc = "Numpy implementation of mnist label predictor."
    parser = argparse.ArgumentParser(description=desc)

    # -- training params
    parser.add_argument('--episodes', type=int, default=3000, help='The number of episodes to run.')

    # -- meta-training params
    parser.add_argument('--steps', type=int, default=5, help='.')  # fixme: add definition
    parser.add_argument('--tasks', type=int, default=5, help='.')  # fixme: add definition
    parser.add_argument('--lr_meta', type=float, default=1e-3, help='.')

    return parser.parse_args()


def main():
    args = parse_args()

    # -- load data
    dataset = OmniglotDataset(steps=args.steps)
    sampler = RandomSampler(data_source=dataset, replacement=True, num_samples=args.episodes * args.tasks)
    dataloader = DataLoader(dataset=dataset, sampler=sampler, batch_size=args.tasks, drop_last=True)  # fixme: what does task mean here?

    # -- train model
    my_train = Train(dataloader, args)
    my_train()


if __name__ == '__main__':
    main()
