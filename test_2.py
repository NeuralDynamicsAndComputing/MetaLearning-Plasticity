import torch
import os
import warnings
import numpy as np
import datetime

from torch import nn, optim
from torch.nn.utils import _stateless
from sklearn.decomposition import PCA
from dataset import EmnistDataset, DataProcess
from torch.utils.data import DataLoader, RandomSampler
from optim import my_optimizer, generic_rule
from torch.nn import functional as F

import matplotlib.pyplot as plt
from utils import log, Plot

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


def stats(model, params, x_qry, y_qry):

    loss_func = nn.CrossEntropyLoss()

    with torch.no_grad():

        # -- compute meta-loss
        _, logits = _stateless.functional_call(model, params, x_qry.unsqueeze(1))
        loss = loss_func(logits, y_qry.reshape(-1)).item()

        # -- compute accuracy
        pred = F.softmax(logits, dim=1).argmax(dim=1)
        accuracy = torch.eq(pred, y_qry.reshape(-1)).sum().item() / len(y_qry.reshape(-1))

    return loss, accuracy


def cumsum_sma(array, period):
    ret = np.cumsum(array, dtype=float)
    ret[period:] = ret[period:] - ret[:-period]

    return ret[period - 1:] / period

# -- initialization
model = MyModel()
opt = 'fix'
print(opt)
if opt == 'fix':
    vec, fbk = [], 'fix'
elif opt == 'fix_16':
    vec, fbk = [16], 'fix'
elif opt == 'fix_12':
    vec, fbk = [12], 'fix'
elif opt == 'sym':
    vec, fbk = [], 'sym'
    model.fk1.weight.data = model.fc1.weight.data
    model.fk2.weight.data = model.fc2.weight.data
    model.fk3.weight.data = model.fc3.weight.data

OptimAdpt = my_optimizer(generic_rule, vec, fbk)

res_dir = os.path.join('./results/trunk/Tests_June_5/Tests', opt, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_'))
os.makedirs(res_dir)

# -- data
K = 50
Q = 10
M = 5
dim = 28
episodes = 300
epochs = 5
dataset = EmnistDataset(K=K, Q=Q, dim=dim)
sampler = RandomSampler(data_source=dataset, replacement=True, num_samples=episodes * M)
dataset = DataLoader(dataset=dataset, sampler=sampler, batch_size=M, drop_last=True)
data_process = DataProcess(K=K, Q=Q, dim=dim)

for eps, data in enumerate(dataset):
    x_trn, y_trn, x_qry, y_qry = data_process(data, M)
    break

torch.save(y_trn, res_dir + '/eps100_y_trn.pt')
torch.save(x_trn, res_dir + '/eps100_x_trn.pt')
torch.save(y_qry, res_dir + '/eps100_y_qry.pt')
torch.save(x_qry, res_dir + '/eps100_x_qry.pt')

# -- model params
params = {key: val.clone() for key, val in dict(model.named_parameters()).items() if '.' in key}
for key in params:
    if 'fc' in key:
        params[key].adapt = True
    else:
        params[key].adapt = False

# -- plasticity params
plasticity_params = nn.ParameterList()
for i in range(19):
    plasticity_params.append(nn.Parameter(torch.tensor(0.).float()))

if opt == 'fix':
    plasticity_params[0] = nn.Parameter(torch.tensor(0.012).float())
    plasticity_params[1] = nn.Parameter(torch.log(torch.tensor(0.).float()))
elif opt == 'fix_16':
    plasticity_params[0] = nn.Parameter(torch.tensor(0.018).float())
    plasticity_params[1] = nn.Parameter(torch.log(torch.tensor(0.).float()))
    plasticity_params[17] = nn.Parameter(torch.log(torch.tensor(-0.075).float()))
    plasticity_params[18] = nn.Parameter(torch.log(torch.tensor(-0.05).float()))
elif opt == 'fix_12':
    plasticity_params[0] = nn.Parameter(torch.tensor(0.017).float())
    plasticity_params[1] = nn.Parameter(torch.log(torch.tensor(0.).float()))
    plasticity_params[11] = nn.Parameter(torch.log(torch.tensor(-0.005).float()))
elif opt == 'sym':
    plasticity_params[0] = nn.Parameter(torch.tensor(0.015).float())
    plasticity_params[1] = nn.Parameter(torch.log(torch.tensor(0.).float()))

Theta = nn.ParameterList(plasticity_params)

# -- train
l, acc = [], []
torch.save(params, res_dir + '/param_eps100_itr{}.pt'.format(0))
for epoch in range(epochs):
    for itr_adapt, (x, label) in enumerate(zip(x_trn, y_trn)):

        # -- stats
        loss, accuracy = stats(model, params, x_qry, y_qry)

        # -- predict
        y, logits = _stateless.functional_call(model, params, x.unsqueeze(0).unsqueeze(0))

        # -- update network params
        angle, e_mean, e_std, e_norm, angle_WB, \
        norm_W, W_mean, W_std, y_mean, y_std, \
        y_norm = OptimAdpt(params, logits, label, y, model.Beta, Theta)

        # -- log
        # log(accuracy, res_dir + '/acc.txt')
        # log(loss, res_dir + '/loss.txt')
        # log([acc], res_dir + '/acc_meta.txt')
        # log([loss.item()], res_dir + '/loss_meta.txt')
        # log(angle, res_dir + '/ang_meta.txt')
        # log(angle_WB, res_dir + '/ang_WB_meta.txt')
        # log(norm_W, res_dir + '/norm_W_meta.txt')
        # log(W_mean, res_dir + '/W_mean_meta.txt')
        # log(W_std, res_dir + '/W_std_meta.txt')
        # log(e_mean, res_dir + '/e_mean_meta.txt')
        # log(e_std, res_dir + '/e_std_meta.txt')
        # log(e_norm, res_dir + '/e_norm_meta.txt')
        # log(y_mean, res_dir + '/y_mean_meta.txt')
        # log(y_std, res_dir + '/y_std_meta.txt')
        # log(y_norm, res_dir + '/y_norm_meta.txt')
        log(angle, res_dir + '/ang.txt')
        # log(W_norms, res_dir + '/norm_W.txt')
        # log(meta_grad, res_dir + '/meta_grad.txt')

    # -- log
    print('Epoch: {}, Batch: {}, Loss: {}, Accuracy: {}'.format(epoch+1, itr_adapt, loss, accuracy))
    torch.save(params, res_dir + '/param_eps100_itr{}.pt'.format(epoch+1))

    l.append(loss)
    acc.append(accuracy)

# -- plot
plt.plot(np.arange(epochs), l)
plt.title('Loss {}'.format(opt))
plt.ylim([0, 4])
plt.savefig(res_dir + '/Loss', bbox_inches='tight')
plt.show()
plt.close()

plt.plot(np.arange(epochs), acc)
plt.title('Accuracy {}'.format(opt))
plt.ylim([0, 1])
plt.savefig(res_dir + '/Accuracy', bbox_inches='tight')
plt.show()
plt.close()

y = np.nan_to_num(np.loadtxt(res_dir + '/ang.txt'))
for idx in range(y.shape[1] - 1):
    # -- moving average
    period = 23
    z = cumsum_sma(y[:, idx], period)
    plt.plot(np.array(range(len(z))) + int((period - 1) / 2), z)


    plt.legend(['1', '2', '3', '4'])

plt.title('Meta Angles for ' + opt)
plt.ylim([30, 100])
plt.savefig(res_dir + '/Angle', bbox_inches='tight')
plt.show()
plt.close()



