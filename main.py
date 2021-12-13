import os
import torch
import warnings
import argparse
import datetime

import numpy as np

from git import Repo
from torch import nn, optim
from torchviz import make_dot
from torch.nn.utils import _stateless
from torch.nn import functional as func
from torch.utils.data import DataLoader, RandomSampler
# from kymatio.torch import Scattering2D

from utils import log
from Dataset import EmnistDataset, OmniglotDataset, DataProcess
from Optim_rule import my_optimizer as OptimAdpt

warnings.simplefilter(action='ignore', category=UserWarning)

np.random.seed(0)
torch.manual_seed(0)


class MyModel(nn.Module):
    def __init__(self, database):
        super(MyModel, self).__init__()

        self.database = database
        if self.database == 'omniglot':
            dim_out = 964
        elif self.database == 'emnist':
            dim_out = 47

        # -- embedding params
        self.cn1 = nn.Conv2d(1, 256, kernel_size=3, stride=2)
        self.cn2 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.cn3 = nn.Conv2d(256, 256, kernel_size=3, stride=2)
        self.cn4 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.cn5 = nn.Conv2d(256, 256, kernel_size=3, stride=2)
        self.cn6 = nn.Conv2d(256, 256, kernel_size=3, stride=2)

        # -- prediction params
        self.fc1 = nn.Linear(2304, 1700)
        self.fc2 = nn.Linear(1700, 1200)
        self.fc3 = nn.Linear(1200, dim_out)

        # -- feedback
        self.fk1 = nn.Linear(2304, 1700, bias=False)
        self.fk2 = nn.Linear(1700, 1200, bias=False)
        self.fk3 = nn.Linear(1200, dim_out, bias=False)

        # -- learning params
        self.alpha = nn.Parameter(torch.rand(1) / 100 - 1)
        self.beta = nn.Parameter(torch.rand(1) / 100 - 1)
        self.alpha_fbk = nn.Parameter(torch.rand(1) / 100 - 1)
        self.beta_fbk = nn.Parameter(torch.rand(1) / 100 - 1)

        # -- non-linearity
        self.relu = nn.ReLU()
        self.Beta = 10
        self.sopl = nn.Softplus(beta=self.Beta)

        # -- learnable params
        self.params = nn.ParameterList()

    def forward(self, x):

        y1 = self.relu(self.cn1(x))
        y2 = self.relu(self.cn2(y1))
        y3 = self.relu(self.cn3(y2))
        y4 = self.relu(self.cn4(y3))
        if self.database == 'omniglot':
            y5 = self.relu(self.cn5(y4))
            y6 = self.relu(self.cn6(y5))
            y6 = y6.view(y6.size(0), -1)
        elif self.database == 'emnist':
            y6 = y4.view(y4.size(0), -1)

        y7 = self.sopl(self.fc1(y6))
        y8 = self.sopl(self.fc2(y7))

        return (y6, y7, y8), self.fc3(y8)


class Train:
    def __init__(self, meta_dataset, args):

        # -- processor params
        self.device = args.device

        # -- data params
        self.database = args.database
        self.meta_dataset = meta_dataset
        self.M = args.M
        self.K = args.K
        self.Q = args.Q
        self.data_process = DataProcess(M=self.M, K=self.K, Q=self.Q, database=self.database, dim=args.dim,
                                        device=self.device)

        # -- model params
        self.path_pretrained = './data/models/omniglot_example/model_stat.pth'
        self.model = self.load_model().to(self.device)
        self.B_init = args.B_init
        # self.scat = Scattering2D(J=3, L=8, shape=(28, 28), max_order=2)

        # -- optimization params
        self.lr_meta = args.lr_meta
        self.loss_func = nn.CrossEntropyLoss()
        self.OptimMeta = optim.Adam(self.model.params.parameters(), lr=self.lr_meta)

        # -- log params
        self.res_dir = args.res_dir

    def load_model(self):
        """
            Loads pretrained parameters for the convolutional layers and sets adaptation and meta training flags for
            parameters.
        """
        # -- init model
        model = MyModel(self.database)
        old_model = torch.load(self.path_pretrained)
        for old_key in old_model:
            dict(model.named_parameters())[old_key].data = old_model[old_key]

        # -- learning flags
        for key, val in model.named_parameters():
            if 'cn' in key:
                val.meta, val.adapt, val.requires_grad = False, False, False
            elif 'fc' in key or 'fk' in key:
                val.meta, val.adapt = False, True
            else:
                val.meta, val.adapt = True, False

            # -- learnable params
            if val.meta is True:
                model.params.append(val)

        return model

    @staticmethod
    def weights_init(m):

        classname = m.__class__.__name__
        if classname.find('Linear') != -1:

            # -- weights
            init_range = torch.sqrt(torch.tensor(6.0 / (m.in_features + m.out_features)))
            m.weight.data.uniform_(-init_range, init_range)

            # -- bias
            if m.bias is not None:
                m.bias.data.uniform_(-init_range, init_range)

    def reinitialize(self):

        self.model.apply(self.weights_init)

        if self.B_init == 'W':  # todo: avoid manually initializing B.
            self.model.fk1.weight.data = self.model.fc1.weight.data
            self.model.fk2.weight.data = self.model.fc2.weight.data
            self.model.fk3.weight.data = self.model.fc3.weight.data

        params = {key: val.clone() for key, val in dict(self.model.named_parameters()).items()}
        for key in params:
            params[key].adapt = dict(self.model.named_parameters())[key].adapt

        return params

    @staticmethod
    def accuracy(logits, label):

        pred = func.softmax(logits, dim=1).argmax(dim=1)

        return torch.eq(pred, label).sum().item() / len(label)

    def stats(self, params, x_qry, y_qry, loss, accuracy):

        with torch.no_grad():

            # -- compute meta-loss
            _, logits = _stateless.functional_call(self.model, params, x_qry.unsqueeze(1))
            loss_meta = self.loss_func(logits, y_qry.reshape(-1))
            loss.append(loss_meta.item())

            # -- compute accuracy
            acc = self.accuracy(logits, y_qry.reshape(-1))
            accuracy.append(acc)

        return loss, accuracy

    def __call__(self):
        """
            Model training.
        """
        self.model.train()
        for eps, data in enumerate(self.meta_dataset):

            # -- initialize
            loss, accuracy = [], []
            params = self.reinitialize()

            # -- training data
            x_trn, y_trn, x_qry, y_qry = self.data_process(data)

            """ adaptation """
            for x, label in zip(x_trn, y_trn):

                # -- stats
                loss, accuracy = self.stats(params, x_qry, y_qry, loss, accuracy)

                # -- predict
                y, logits = _stateless.functional_call(self.model, params, x.unsqueeze(0).unsqueeze(0))

                if False:
                    make_dot(logits, params=dict(list(self.model.named_parameters()))).render('comp_grph', format='png')
                    quit()

                # -- update network params
                params = OptimAdpt(params, logits, label, y, self.model.Beta, self.model.alpha,
                                              self.model.beta, self.model.alpha_fbk, self.model.beta_fbk)

            """ meta update """
            # -- predict
            _, logits = _stateless.functional_call(self.model, params, x_qry.unsqueeze(1))
            if False:
                make_dot(logits, params=dict(list(self.model.named_parameters()))).render('comp_grph', format='png')
                quit()

            # -- compute loss and accuracy
            loss_meta = self.loss_func(logits, y_qry.reshape(-1))
            acc = self.accuracy(logits, y_qry.reshape(-1))

            # -- update params
            self.OptimMeta.zero_grad()
            loss_meta.backward()
            self.OptimMeta.step()

            # -- log
            log(accuracy, self.res_dir + '/acc.txt')
            log(loss, self.res_dir + '/loss.txt')
            log([acc], self.res_dir + '/acc_meta.txt')
            log([loss_meta.item()], self.res_dir + '/loss_meta.txt')

            print('Train Episode: {}\tLoss: {:.6f}\tAccuracy: {:.3f}\tlr (W): {:.6f}\tdr (W): {:.6f}'
                  '\tlr (B): {:.6f}\tdr (B): {:.6f}'.format(eps+1, loss_meta.item(), acc,
                                                    torch.exp(self.model.alpha).detach().cpu().numpy()[0],
                                                    torch.exp(self.model.beta).detach().cpu().numpy()[0],
                                                    torch.exp(self.model.alpha_fbk).detach().cpu().numpy()[0],
                                                    torch.exp(self.model.beta_fbk).detach().cpu().numpy()[0]))


def parse_args():
    desc = "Pytorch implementation of meta-plasticity model."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gpu_mode', type=int, default=1, help='Accelerate the script using GPU.')

    # -- data params
    parser.add_argument('--database', type=str, default='emnist', help='The database.')
    parser.add_argument('--dim', type=int, default=28, help='The dimension of the training data.')

    # -- meta-training params
    parser.add_argument('--episodes', type=int, default=3000, help='The number of training episodes.')
    parser.add_argument('--K', type=int, default=20, help='The number of training datapoints per class.')
    parser.add_argument('--Q', type=int, default=5, help='The number of query datapoints per class.')
    parser.add_argument('--M', type=int, default=5, help='The number of classes per task.')
    parser.add_argument('--lr_meta', type=float, default=5e-2, help='.')

    # -- log params
    parser.add_argument('--res', type=str, default='results', help='Path for storing the results.')

    # -- model params
    parser.add_argument('--B_init', type=str, default='W',
                        help='Feedback initial value: 1) B_init = rand; 2) B_init = W^T.')

    args = parser.parse_args()

    # -- storage settings
    s_dir = os.getcwd()
    local_repo = Repo(path=s_dir)
    args.res_dir = os.path.join(s_dir, args.res, local_repo.active_branch.name,
                                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(args.res_dir)

    # -- GPU settings
    args.device = torch.device('cuda' if (bool(args.gpu_mode) and torch.cuda.is_available()) else 'cpu')

    return check_args(args)


def check_args(args):
    # todo: Implement argument check.
    if bool(args.gpu_mode) and not torch.cuda.is_available():
        print('No GPUs on this device! Running on CPU.')

    # -- store settings
    with open(args.res_dir + '/args.txt', 'w') as fp:
        for item in vars(args).items():
            fp.write("{} : {}\n".format(item[0], item[1]))

    return args


def main():
    args = parse_args()

    # -- load data
    if args.database == 'emnist':
        dataset = EmnistDataset(K=args.K, Q=args.Q)
    elif args.database == 'omniglot':
        dataset = OmniglotDataset(K=args.K, Q=args.Q, dim=args.dim)
    sampler = RandomSampler(data_source=dataset, replacement=True, num_samples=args.episodes * args.M)
    meta_dataset = DataLoader(dataset=dataset, sampler=sampler, batch_size=args.M, drop_last=True)

    # -- train model
    my_train = Train(meta_dataset, args)
    my_train()


if __name__ == '__main__':
    main()
