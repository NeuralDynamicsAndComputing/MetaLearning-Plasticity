import os
import torch
import warnings
import argparse
import datetime

from torch import nn, optim
from random import randrange
from torch.nn.utils import _stateless
from torch.utils.data import DataLoader, RandomSampler

from utils import log, Plot, meta_stats
from optim import my_optimizer, generic_rule
from dataset import EmnistDataset, DataProcess

warnings.simplefilter(action='ignore', category=UserWarning)


class MyModel(nn.Module):
    """
        Classifier model

    Defines the layers and parameters of
        1) the classification network,
        2) feedback alignment model, and
        3) the plasticity meta-parameters.
    """
    def __init__(self, args):
        super(MyModel, self).__init__()

        # -- forward pathway
        dim_out = 47
        self.fc1 = nn.Linear(784, 170, bias=False)
        self.fc2 = nn.Linear(170, 130, bias=False)
        self.fc3 = nn.Linear(130, 100, bias=False)
        self.fc4 = nn.Linear(100, 70, bias=False)
        self.fc5 = nn.Linear(70, dim_out, bias=False)

        # -- feedback pathway
        self.fk1 = nn.Linear(784, 170, bias=False)
        self.fk2 = nn.Linear(170, 130, bias=False)
        self.fk3 = nn.Linear(130, 100, bias=False)
        self.fk4 = nn.Linear(100, 70, bias=False)
        self.fk5 = nn.Linear(70, dim_out, bias=False)

        # -- plasticity params
        self.alpha_fbk = nn.Parameter(torch.rand(1) / 100 - 1)
        self.beta_fbk = nn.Parameter(torch.rand(1) / 100 - 1)
        self.a_fwd = nn.Parameter(torch.tensor(args.a).float())
        self.b_fwd = nn.Parameter(torch.tensor(args.b).float())
        self.c_fwd = nn.Parameter(torch.tensor(args.c).float())
        self.d_fwd = nn.Parameter(torch.tensor(args.d).float())
        self.e_fwd = nn.Parameter(torch.tensor(args.e).float())
        self.f_fwd = nn.Parameter(torch.tensor(args.f).float())
        self.g_fwd = nn.Parameter(torch.tensor(args.g).float())
        self.h_fwd = nn.Parameter(torch.tensor(args.h).float())
        self.i_fwd = nn.Parameter(torch.tensor(args.i).float())
        self.j_fwd = nn.Parameter(torch.tensor(args.i).float())

        # -- activation function
        self.Beta = 10
        self.sopl = nn.Softplus(beta=self.Beta)

        # -- meta-params
        self.params_fwd = nn.ParameterList()
        self.params_fbk = nn.ParameterList()

    def forward(self, x):
        """
            performs forward pass of information for the classification network.
        :param x: input images,
        :return: input, activations across network layers, and predicted output.
        """
        y0 = x.squeeze(1)

        y1 = self.sopl(self.fc1(y0))
        y2 = self.sopl(self.fc2(y1))
        y3 = self.sopl(self.fc3(y2))
        y4 = self.sopl(self.fc4(y3))

        return (y0, y1, y2, y3, y4), self.fc5(y4)


class MetaLearner:
    def __init__(self, metatrain_dataset, args):

        # -- processor params
        self.device = args.device

        # -- data params
        self.K = args.K
        self.M = args.M
        self.database = args.database
        self.metatrain_dataset = metatrain_dataset
        self.data_process = DataProcess(K=self.K, Q=args.Q, dim=args.dim, device=self.device)

        # -- model params
        self.evl = args.evl
        self.model = self.load_model(args).to(self.device)
        self.Theta = nn.ParameterList([*self.model.params_fwd, *self.model.params_fbk])
        self.fbk = args.fbk

        # -- optimization params
        self.lamb = args.lamb
        self.loss_func = nn.CrossEntropyLoss()
        self.OptimAdpt = my_optimizer(generic_rule, args.vec, args.fbk)
        self.OptimMeta = optim.Adam([{'params': self.model.params_fwd.parameters(), 'lr': args.lr_meta_fwd},
                                     {'params': self.model.params_fbk.parameters(), 'lr': args.lr_meta_fbk}])

        # -- log params
        self.res_dir = args.res_dir
        self.plot = Plot(self.res_dir)

    def load_model(self, args):
        """
            Loads pretrained parameters for the convolutional layers and sets adaptation and meta training flags for
            parameters.
        """
        # -- init model
        model = MyModel(args)

        # -- learning flags
        for key, val in model.named_parameters():
            if 'fc' in key:
                val.meta_fwd, val.meta_fbk, val.adapt = False, False, True
            elif 'fk' in key:
                if self.evl:
                    val.meta_fwd, val.meta_fbk, val.adapt = False, False, True
                else:
                    val.meta_fwd, val.meta_fbk, val.adapt, val.requires_grad = False, False, False, False
            elif 'fwd' in key:
                val.meta_fwd, val.meta_fbk, val.adapt = True, False, False
            elif 'fbk' in key:
                if self.evl:
                    val.meta_fwd, val.meta_fbk, val.adapt = False, True, False
                else:
                    val.meta_fwd, val.meta_fbk, val.adapt = False, False, False

            # -- meta-params
            if val.meta_fwd is True:
                model.params_fwd.append(val)
            elif val.meta_fbk is True:
                model.params_fbk.append(val)

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

        if self.fbk == 'sym':
            self.model.fk1.weight.data = self.model.fc1.weight.data
            self.model.fk2.weight.data = self.model.fc2.weight.data
            self.model.fk3.weight.data = self.model.fc3.weight.data
            self.model.fk4.weight.data = self.model.fc4.weight.data
            self.model.fk5.weight.data = self.model.fc5.weight.data

        params = {key: val.clone() for key, val in dict(self.model.named_parameters()).items() if '.' in key}
        for key in params:
            params[key].adapt = dict(self.model.named_parameters())[key].adapt

        return params

    def train(self):
        """
            Model training.
        """
        self.model.train()
        for eps, data in enumerate(self.metatrain_dataset):

            # -- initialize
            params = self.reinitialize()

            # -- training data
            x_trn, y_trn, x_qry, y_qry = self.data_process(data, self.M)

            """ adaptation """
            for itr_adapt, (x, label) in enumerate(zip(x_trn, y_trn)):

                # -- predict
                y, logits = _stateless.functional_call(self.model, params, x.unsqueeze(0).unsqueeze(0))

                # -- update network params
                self.OptimAdpt(params, logits, label, y, self.model.Beta, self.Theta)

            """ meta update """
            # -- predict
            y, logits = _stateless.functional_call(self.model, params, x_qry.unsqueeze(1))

            # -- L1 regularization
            l1_reg = None
            for T in self.model.params_fwd.parameters():
                if l1_reg is None:
                    l1_reg = T.norm(1)
                else:
                    l1_reg = l1_reg + T.norm(1)

            loss_meta = self.loss_func(logits, y_qry.ravel()) + l1_reg * self.lamb

            # -- compute and store meta stats
            acc = meta_stats(logits, params, y_qry.ravel(), y, self.model.Beta, self.res_dir)

            # -- update params
            Theta = [p.detach().clone() for p in self.Theta]
            self.OptimMeta.zero_grad()
            loss_meta.backward()
            self.OptimMeta.step()

            # -- log
            log([loss_meta.item()], self.res_dir + '/loss_meta.txt')

            line = 'Train Episode: {}\tLoss: {:.6f}\tAccuracy: {:.3f}'.format(eps+1, loss_meta.item(), acc)
            for idx, param in enumerate(Theta):
                line += '\tMetaParam_{}: {:.6f}'.format(idx + 1, param.cpu().numpy())
            print(line)
            with open(self.res_dir + '/params.txt', 'a') as f:
                f.writelines(line+'\n')

        # -- plot
        self.plot()

    def test(self, metatest_dataset, name, M):
        """
            Meta testing.
        """
        self.model.train()

        idx_plot = []
        for eps, data in enumerate(metatest_dataset):
            # -- initialize
            loss, accuracy = [], []
            params = self.reinitialize()

            # -- training data
            x_trn, y_trn, x_qry, y_qry = self.data_process(data, M)

            """ train """
            for itr_adapt, (x, label) in enumerate(zip(x_trn, y_trn)):

                # -- stats
                loss, accuracy = self.stats(params, x_qry, y_qry, loss, accuracy)

                # -- predict
                y, logits = _stateless.functional_call(self.model, params, x.unsqueeze(0).unsqueeze(0))

                # -- update network params
                _ = self.OptimAdpt(params, logits, label, y, self.model.Beta, self.Theta)

            # -- compute loss and accuracy
            loss, accuracy = self.stats(params, x_qry, y_qry, loss, accuracy)

            # -- log
            log(accuracy, self.res_dir + '/acc_test_' + name + '.txt')
            log(loss, self.res_dir + '/loss_test_' + name + '.txt')

            idx_plot.append(eps)

        # -- plot
        self.plot.adapt_accuracy(filename='/acc_test_' + name + '.txt', savename='/adapt_accuracy_' + name, idx_plot=idx_plot)
        self.plot.adapt_loss(filename='/loss_test_' + name + '.txt', savename='/adapt_loss_' + name, idx_plot=idx_plot)


def parse_args():
    desc = "Pytorch implementation of meta-plasticity model."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gpu_mode', type=int, default=1, help='Accelerate the script using GPU.')
    parser.add_argument('--seed', type=int, default=1, help='.')

    # -- data params
    parser.add_argument('--database', type=str, default='emnist', help='The database.')
    parser.add_argument('--dim', type=int, default=28, help='The dimension of the training data.')
    parser.add_argument('--test_name', type=str, default='', help='Test folder name.')

    # -- meta-training params
    parser.add_argument('--episodes', type=int, default=10002, help='The number of training episodes.')
    parser.add_argument('--K', type=int, default=50, help='The number of training datapoints per class.')
    parser.add_argument('--Q', type=int, default=5, help='The number of query datapoints per class.')
    parser.add_argument('--M', type=int, default=5, help='The number of classes per task.')
    parser.add_argument('--lamb', type=float, default=1.5, help='.')
    parser.add_argument('--lr_meta_fwd', type=float, default=5e-3, help='.')
    parser.add_argument('--lr_meta_fbk', type=float, default=5e-3, help='.')
    parser.add_argument('--a', type=float, default=5e-3, help='.')
    parser.add_argument('--b', type=float, default=0., help='.')
    parser.add_argument('--c', type=float, default=0., help='.')
    parser.add_argument('--d', type=float, default=0., help='.')
    parser.add_argument('--e', type=float, default=0., help='.')
    parser.add_argument('--f', type=float, default=0., help='.')
    parser.add_argument('--g', type=float, default=0., help='.')
    parser.add_argument('--h', type=float, default=0., help='.')
    parser.add_argument('--i', type=float, default=0., help='.')

    # -- log params
    parser.add_argument('--res', type=str, default='results', help='Path for storing the results.')

    # -- model params
    parser.add_argument('--vec', nargs='*', default=[], help='Learning rule terms.')
    parser.add_argument('--fbk', type=str, default='sym',
                        help='Feedback matrix type: 1) sym = Symmetric matrix; 2) fix = Fixed random matrix.')

    args = parser.parse_args()

    # -- storage settings
    s_dir = os.getcwd()
    args.res_dir = os.path.join(s_dir, args.res, args.test_name,
                                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str(randrange(40)))
    os.makedirs(args.res_dir)

    # -- GPU settings
    args.device = torch.device('cuda' if (bool(args.gpu_mode) and torch.cuda.is_available()) else 'cpu')

    # -- feedback type
    args.evl = False

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

    # -- meta-train
    dataset = EmnistDataset(K=args.K, Q=args.Q, dim=args.dim)
    sampler = RandomSampler(data_source=dataset, replacement=True, num_samples=args.episodes * args.M)
    metatrain_dataset = DataLoader(dataset=dataset, sampler=sampler, batch_size=args.M, drop_last=True)
    metaplasticity_model = MetaLearner(metatrain_dataset, args)
    metaplasticity_model.train()


if __name__ == '__main__':
    main()
