import os
import torch
import warnings
import argparse
import datetime

import numpy as np

from git import Repo
from torch import nn, optim
from random import randrange
from torchviz import make_dot
from torch.nn.utils import _stateless
from sklearn.decomposition import PCA
from torch.nn import functional as func
# from GPUtil import showUtilization as gpu_usage
from torch.utils.data import DataLoader, RandomSampler
import matplotlib.pyplot as plt
from utils import log, Plot
from dataset import MNISTDataset, EmnistDataset, FashionMNISTDataset, OmniglotDataset, DataProcess
from optim import my_optimizer, evolve_rule, generic_rule

warnings.simplefilter(action='ignore', category=UserWarning)

np.random.seed(6)
torch.manual_seed(6)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # -- prediction params
        dim_out = 47
        self.fc1 = nn.Linear(784, 130, bias=False)
        self.fc2 = nn.Linear(130, 70, bias=False)
        self.fc3 = nn.Linear(70, dim_out, bias=False)

        # -- feedback
        self.fk1 = nn.Linear(784, 130, bias=False)
        self.fk2 = nn.Linear(130, 70, bias=False)
        self.fk3 = nn.Linear(70, dim_out, bias=False)

        # -- non-linearity
        self.relu = nn.ReLU()
        self.Beta = 10
        self.sopl = nn.Softplus(beta=self.Beta)

    def forward(self, x):

        y0 = x.squeeze(1)

        y1 = self.sopl(self.fc1(y0))
        y2 = self.sopl(self.fc2(y1))

        return (y0, y1, y2), self.fc3(y2)


model = MyModel()
weight = 'fc1.weight'
plasticity_rule = 'SYM'  # FIX, FIX_12, FIX_16, SYM

if weight == 'fc1.weight':
    W_d1, W_d2 = 130, 784
elif weight == 'fc2.weight':
    W_d1, W_d2 = 70, 130
elif weight == 'fc3.weight':
    W_d1, W_d2 = 47, 70

if plasticity_rule == 'FIX':
    res_dir = './results/trunk/Tests_May_32/Tests/FIX/2022-06-01_16-44-49_26/'
if plasticity_rule == 'FIX_12':
    res_dir = './results/trunk/Tests_May_32/Tests/FIX_12/2022-06-01_16-44-58_11/'
if plasticity_rule == 'FIX_16':
    res_dir = './results/trunk/Tests_May_32/Tests/FIX_16/2022-06-01_16-45-08_28/'
if plasticity_rule == 'SYM':
    res_dir = './results/trunk/Tests_May_32/Tests/SYM/2022-06-01_16-44-28_11/'

eps = 100
loss_func = nn.CrossEntropyLoss()
params = torch.load(res_dir + 'param_eps{}_itr249.pt'.format(eps))
W_n = params[weight].reshape(1, -1)

# -- load data
x_trn = torch.load(res_dir + 'eps{}_x_trn.pt'.format(eps))
y_trn = torch.load(res_dir + 'eps{}_y_trn.pt'.format(eps))
x_qry = torch.load(res_dir + 'eps{}_x_qry.pt'.format(eps))
y_qry = torch.load(res_dir + 'eps{}_y_qry.pt'.format(eps))

# -- construct matrix M
for itr_adapt in range(249):
    W = torch.load(res_dir + 'param_eps{}_itr{}.pt'.format(eps, itr_adapt))[weight].reshape(1, -1) - W_n

    try:
        M = torch.cat((M, W))
    except:
        M = W.clone()

# -- find directions
pca = PCA(n_components=2)
pca.fit(M.cpu().detach().numpy())
delta = pca.components_

# -- construct grid
plt.figure(0)
fig, (ax1, ax2) = plt.subplots(ncols=2)

n = 40
x_min, x_max, y_min, y_max = -5, 0, -1, 1
alph, beta = torch.meshgrid([torch.linspace(x_min, x_max, steps=n), torch.linspace(y_min, y_max, steps=n)])
loss = []
for i in range(len(alph)):
    for j in range(len(beta)):

        W3_grid = (W_n + alph[i, j] * delta[0] + beta[i, j] * delta[1]).reshape(W_d1, W_d2)
        params[weight].data = W3_grid.data

        # -- compute loss
        _, logits = _stateless.functional_call(model, params, x_qry.unsqueeze(1))
        loss.append(loss_func(logits, y_qry.reshape(-1)).item())

X = alph.numpy()
Y = beta.numpy()
Z = np.array(loss).reshape(n, n)
ax1.pcolor(X, Y, Z)

cs = ax2.tricontour(X.ravel(), Y.ravel(), Z.ravel(), levels=14, linewidths=0.5)
ax2.clabel(cs)
ax2.tricontourf(X.ravel(), Y.ravel(), Z.ravel(), levels=14, alpha=0.3)

# -- vis trajectory
for itr_adapt in range(249):
    params = torch.load(res_dir + 'param_eps{}_itr{}.pt'.format(eps, itr_adapt))[weight].reshape(1, -1)

    traj = pca.transform(params.cpu().detach().numpy())

    ax1.scatter(traj[0, 0], traj[0, 1], s=2, c='r')
    ax2.scatter(traj[0, 0], traj[0, 1], s=2, c='r')

ax1.scatter(traj[0, 0], traj[0, 1], s=40, c='r', marker='x')
ax2.scatter(traj[0, 0], traj[0, 1], s=40, c='r', marker='x')

# -- plot
plt.title('Loss landscape for {} with {}'.format(weight, plasticity_rule))
plt.show()
