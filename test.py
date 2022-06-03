import torch
import warnings
import numpy as np

from torch import nn, optim
from torch.nn.utils import _stateless
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=UserWarning)

np.random.seed(0)
torch.manual_seed(0)


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


def w2vec(param_dir):

    params = torch.load(param_dir)

    W_1 = params['fc1.weight'].reshape(1, -1)
    W_2 = params['fc2.weight'].reshape(1, -1)
    W_3 = params['fc3.weight'].reshape(1, -1)

    return torch.cat((W_1, W_2, W_3), dim=1)

def vec2w(theta):

    params = {}

    params['fc1.weight'] = theta[:, :(130 * 784)].reshape(130, 784)
    params['fc2.weight'] = theta[:, (130 * 784):(130 * 784 + 70 * 130)].reshape(70, 130)
    params['fc3.weight'] = theta[:, (130 * 784 + 70 * 130):].reshape(47, 70)

    return params

model = MyModel()
plasticity_rule = 'SYM'  # FIX, FIX_12, FIX_16, SYM

if plasticity_rule == 'FIX':
    res_dir = './results/trunk/Tests_May_32/Tests/FIX/2022-06-01_16-44-49_26/'
if plasticity_rule == 'FIX_12':
    res_dir = './results/trunk/Tests_May_32/Tests/FIX_12/2022-06-01_16-44-58_11/'
if plasticity_rule == 'FIX_16':
    res_dir = './results/trunk/Tests_May_32/Tests/FIX_16/2022-06-01_16-45-08_28/'
if plasticity_rule == 'SYM':
    res_dir = './results/trunk/Tests_May_32/Tests/SYM/2022-06-01_16-44-28_11/'

eps = 100
n = 248
loss_func = nn.CrossEntropyLoss()
theta_n = w2vec(res_dir + 'param_eps{}_itr{}.pt'.format(eps, n))

# -- load data
x_trn = torch.load(res_dir + 'eps{}_x_trn.pt'.format(eps))
y_trn = torch.load(res_dir + 'eps{}_y_trn.pt'.format(eps))
x_qry = torch.load(res_dir + 'eps{}_x_qry.pt'.format(eps))
y_qry = torch.load(res_dir + 'eps{}_y_qry.pt'.format(eps))

# -- construct matrix M
for i in range(n):
    theta_i = w2vec(res_dir + 'param_eps{}_itr{}.pt'.format(eps, i)) - theta_n

    try:
        M = torch.cat((M, theta_i))
    except:
        M = theta_i.clone()

# -- find directions
pca = PCA(n_components=2)
pca.fit(M.cpu().detach().numpy())
delta = pca.components_

# -- construct grid
fig, (ax1, ax2) = plt.subplots(ncols=2)

n_x, n_y = 51, 51
x_min, x_max, y_min, y_max = -2, 9, -2.5, 0.75
alph, beta = torch.meshgrid([torch.linspace(x_min, x_max, steps=n_x), torch.linspace(y_min, y_max, steps=n_y)])
loss = []

X = alph.numpy()
Y = beta.numpy()
Z = np.zeros_like(X)

for i in range(len(alph)):
    for j in range(len(beta)):

        params = vec2w(theta_n + alph[i, j] * delta[0] + beta[i, j] * delta[1])

        # -- compute loss
        _, logits = _stateless.functional_call(model, params, x_trn.unsqueeze(1))
        Z[i, j] = loss_func(logits, y_trn.reshape(-1)).item()

ax1.pcolor(X, Y, Z)

cs = ax2.tricontour(X.ravel(), Y.ravel(), Z.ravel(), levels=20, linewidths=0.5)
ax2.clabel(cs)
ax2.tricontourf(X.ravel(), Y.ravel(), Z.ravel(), levels=20, alpha=0.3)

# -- vis trajectory
for itr_adapt in range(n+1):
    params = torch.load(res_dir + 'param_eps{}_itr{}.pt'.format(eps, itr_adapt))
    _, logits = _stateless.functional_call(model, params, x_trn.unsqueeze(1))
    print(loss_func(logits, y_trn.reshape(-1)).item())

    theta = w2vec(res_dir + 'param_eps{}_itr{}.pt'.format(eps, itr_adapt))

    traj = np.matmul((theta-theta_n).cpu().detach().numpy(), delta.T)

    ax1.scatter(traj[0, 0], traj[0, 1], s=2, c='r')
    ax2.scatter(traj[0, 0], traj[0, 1], s=2, c='r')

ax1.scatter(traj[0, 0], traj[0, 1], s=40, c='r', marker='x')
ax2.scatter(traj[0, 0], traj[0, 1], s=40, c='r', marker='x')

# -- plot
plt.suptitle('Loss landscape for {}'.format(plasticity_rule))
plt.show()
