import os
import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from torch.nn.utils import _stateless
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

from utils import log, measure_angle


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # -- prediction params
        dim_out = 10

        self.fc1 = nn.Linear(784, 170, bias=False)
        self.fc2 = nn.Linear(170, 130, bias=False)
        self.fc3 = nn.Linear(130, 100, bias=False)
        self.fc4 = nn.Linear(100, 70, bias=False)
        self.fc5 = nn.Linear(70, dim_out, bias=False)

        # -- feedback
        self.fk1 = nn.Linear(784, 170, bias=False)
        self.fk2 = nn.Linear(170, 130, bias=False)
        self.fk3 = nn.Linear(130, 100, bias=False)
        self.fk4 = nn.Linear(100, 70, bias=False)
        self.fk5 = nn.Linear(70, dim_out, bias=False)

        # -- non-linearity
        self.Beta = 10
        self.sopl = nn.Softplus(beta=self.Beta)

    def forward(self, x):

        y0 = torch.cat(torch.unbind(x.squeeze(1), dim=-1), dim=-1)

        y1 = self.sopl(self.fc1(y0))
        y2 = self.sopl(self.fc2(y1))
        y3 = self.sopl(self.fc3(y2))
        y4 = self.sopl(self.fc4(y3))
        return (y0, y1, y2, y3, y4), self.fc5(y4)


def generic_rule(activation, e, params, feedback, vec, fbk):

    lr, t12, t16, t16b = 0.0035, -0.0, -0.015, -.0100
    # FA+12: lr, t12, t16, t16b = lr, t12, t16, t16b = 0.01, -0.0040, 0., 0.
    # FA: lr, t12, t16, t16b = 0.0029, 0., 0., 0.
    # DFA+12: lr, t12, t16, t16b = 0.007, -0.0025, 0., 0.
    # DFA: lr, t12, t16, t16b = 0.0021, 0., 0., 0.
    # DFA+16: 0.0031, -0.0, -0.015, -0.001
    # -- weight update
    i = 0
    for k, p in params.items():
        if 'fc' in k:
            p.update = - lr * torch.matmul(e[i + 1].T, activation[i])

            if '12' in vec:
                p.update -= t12 * (torch.matmul(activation[i + 1].T, activation[i]) - torch.matmul(torch.matmul(activation[i + 1].T, activation[i + 1]), p))
            if '16' in vec:
                p.update -= t16 * torch.matmul(e[i + 1].T, (e[i] - t16b))

            params[k] = p + p.update

            i += 1

    if fbk == 'sym':
        # -- feedback update (symmetric)
        feedback_ = dict({k: v for k, v in params.items() if 'fc' in k})
        for i, ((k, B), (k_, _)) in enumerate(zip(feedback.items(), feedback_.items())):
            params[k].data = params[k_]


class MyOptimizer:
    def __init__(self, update_rule, vec, fbk):
        self.update_rule = update_rule
        self.vec = vec
        self.fbk = fbk

    def __call__(self, params, logits, label, activation, Beta):

        """
            One step update of the inner-loop (derived formulation).
        :param params: model parameters
        :param logits: unnormalized prediction values
        :param label: target class
        :param activation: vector of activations
        :param Beta: smoothness coefficient for non-linearity
        :param Theta: meta-parameters

        :return:
        """
        # -- error
        feedback = dict({k: v for k, v in params.items() if 'fk' in k})
        e = [F.softmax(logits) - F.one_hot(label, num_classes=10)]
        for y, i in zip(reversed(activation), reversed(list(feedback))):
            e.insert(0, torch.matmul(e[0], feedback[i]) * (1 - torch.exp(-Beta * y)))

        # -- weight update
        self.update_rule([*activation, F.softmax(logits, dim=1)], e, params, feedback, self.vec, self.fbk)


class Train:
    def __init__(self, vec, fbk, res_dir, test_name):
        dim = 28
        transform = transforms.Compose([transforms.Resize((dim, dim)), transforms.ToTensor()])
        # transforms.Normalize((0.1307,), (0.3081,))

        np.random.seed(5)
        torch.manual_seed(5)

        dataset_train = datasets.MNIST('../../../data', train=True, download=False, transform=transform)
        self.train_loader = DataLoader(dataset_train, batch_size=1, shuffle=True)

        n_test = 100
        dataset_test = datasets.MNIST('../../../data', train=False, download=False, transform=transform)
        test_sampler = SubsetRandomSampler(np.random.choice(range(10000), n_test, False))
        self.test_loader = DataLoader(dataset_test, batch_size=n_test, sampler=test_sampler)

        self.model = MyModel()
        self.fbk = fbk

        self.OptimAdpt = MyOptimizer(generic_rule, vec, self.fbk)

        self.loss_func = nn.CrossEntropyLoss()

        self.res_dir = res_dir
        self.test_name = test_name

    @staticmethod
    def weights_init(m):

        classname = m.__class__.__name__
        if classname.find('Linear') != -1:

            # -- weights
            init_range = torch.sqrt(torch.tensor(6.0 / (m.in_features + m.out_features)))
            m.weight.data.uniform_(-init_range, init_range)

    def reinitialize(self):

        self.model.apply(self.weights_init)

        if self.fbk == 'sym':
            self.model.fk1.weight.data = self.model.fc1.weight.data
            self.model.fk2.weight.data = self.model.fc2.weight.data
            try:
                self.model.fk3.weight.data = self.model.fc3.weight.data
                self.model.fk4.weight.data = self.model.fc4.weight.data
                self.model.fk5.weight.data = self.model.fc5.weight.data
            except:
                pass

        params = {key: val.clone() for key, val in dict(self.model.named_parameters()).items() if '.' in key}

        return params

    def stats(self, params, Beta):

        for x, label in self.test_loader:

            with torch.no_grad():

                # -- predict
                y, logits = _stateless.functional_call(self.model, params, x.squeeze(1))

                # -- accuracy
                pred = F.softmax(logits, dim=1).argmax(dim=1)
                acc = torch.eq(pred, label).sum().item() / len(label)

                # -- loss
                loss = self.loss_func(logits, label.ravel())

                # -- activation stats
                y_norm, y_mean, y_std = [], [], []
                for y_ in [*y, F.softmax(logits, dim=1)]:
                    y_norm.append(y_.norm(dim=1).mean().item())
                    y_mean.append(y_.mean().item())
                    y_std.append(y_.std(dim=1).mean().item())

                log(y_norm, self.res_dir + '/y_norm_meta.txt')
                log(y_mean, self.res_dir + '/y_mean_meta.txt')
                log(y_std, self.res_dir + '/y_std_meta.txt')

                # -- modulator vector stats
                feedback = dict({k: v for k, v in params.items() if 'fk' in k})
                e = [F.softmax(logits) - F.one_hot(label, num_classes=10)]
                for y_, i in zip(reversed(y), reversed(list(feedback))):
                    e.insert(0, torch.matmul(e[0], feedback[i]) * (1 - torch.exp(-Beta * y_)))

                e_norm, e_mean, e_std = [], [], []
                for e_ in e:
                    e_norm.append(e_.norm(dim=1).mean().item())
                    e_mean.append(e_.mean().item())
                    e_std.append(e_.std(dim=1).mean().item())

                e_sym = [e[-1]]
                W = dict({k: v for k, v in params.items() if 'fc' in k})
                for y_, i in zip(reversed(y), reversed(list(W))):
                    e_sym.insert(0, torch.matmul(e_sym[0], W[i]) * (1 - torch.exp(-Beta * y_)))

                # -- angle b/w modulator vectors e_FA and e_BP
                e_angl = []
                for e_fix_, e_sym_ in zip(e, e_sym):
                    e_angl.append(measure_angle(e_fix_.mean(dim=0), e_sym_.mean(dim=0)))

                log(e_norm, res_dir + '/e_norm_meta.txt')
                log(e_mean, res_dir + '/e_mean_meta.txt')
                log(e_std, res_dir + '/e_std_meta.txt')
                log(e_angl, res_dir + '/e_ang_meta.txt')

                # -- orthonormality
                W = dict({k: v for k, v in params.items() if 'fc' in k})
                ort = []
                for i, w in enumerate(W.values()):
                    ort.append((y[i] - torch.matmul(torch.matmul(y[i], w.T), w)).norm(dim=1).mean()**2)

                log(ort, res_dir + '/ort.txt')

        return loss, acc

    def train(self):

        self.model.train()

        # -- initialize
        params = self.reinitialize()

        for eps, (x, label) in enumerate(self.train_loader):

            # -- predict
            y, logits = _stateless.functional_call(self.model, params, x)

            # -- update network params
            self.OptimAdpt(params, logits, label, y, self.model.Beta)

            # -- stats
            loss, acc = self.stats(params, self.model.Beta)

            print('Iteration {}: loss = {}, acc = {} '.format(eps, loss, acc))

            log([loss], self.res_dir + '/loss.txt')
            log([acc], self.res_dir + '/acc.txt')


fbk = 'fix'
vec = ['16']
test_name = 'fix_FA_16_d/'
res_dir = './tests/' + test_name
if not os.path.exists(res_dir):
    os.mkdir(res_dir)
my_train = Train(vec, fbk, res_dir, test_name)
my_train.train()
