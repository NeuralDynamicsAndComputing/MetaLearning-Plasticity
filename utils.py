import os
import re
import torch

import numpy as np
import matplotlib.pyplot as plt

from torch.nn import functional as F


class Plot:
    def __init__(self, res_dir, meta_param_size):  # todo: pass period as argument
        self.res_dir = res_dir
        self.period = 11
    def comp_moving_avg(self, vector, period):
        """
            Compute moving average.

        The function computes a moving average for the input data vector over
        a given window size. It does so by first calculating the cumulative sum
        of the data vector using numpy's cumsum function. It then subtracts the
        cumulative sum of the data vector up to (period - 1)th index from the
        cumulative sum of the data vector starting from the period-th index.
        Finally, it divides the result by the window size to obtain the moving
        average vector.

        :param vector: input data,
        :param period: window size,
        :return: a vector of moving average values computed using the input data
            and window size
        """
        ret = np.cumsum(vector, dtype=float)
        ret[period:] = ret[period:] - ret[:-period]

        return ret[period - 1:] / period

    def meta_accuracy(self):
        """
            meta accuracy
        """
        # -- compute moving average
        z = self.comp_moving_avg(np.nan_to_num(np.loadtxt(self.res_dir + '/acc_meta.txt')), self.period)

        # -- plot
        plt.plot(np.array(range(len(z))) + int((self.period - 1) / 2), z)

        plt.title('Meta Accuracy')
        plt.ylim([0, 1])
        # plt.xlim([0, self.N])
        plt.savefig(self.res_dir + '/meta_accuracy', bbox_inches='tight')
        plt.close()

    def meta_parameters(self):
        """
            meta parameters
        """
        # fixme: read 'vec' and use for plotting and legend
        # -- read meta params
        with open(self.res_dir + '/params.txt', 'r') as f:
            strings = re.findall(r'(-?\d+\.\d+|nan)', f.read())

        y = np.nan_to_num(np.asarray([float(i) for i in strings])).reshape(-1, 21)
        meta_param_lr_dr = y[:, 2:4]
        meta_param_terms = y[:, 4:]

        # -- plot meta params
        my_legend = ['BP']
        cmap = plt.get_cmap("tab10")

        # -- backprop term
        plt.plot(range(len(y)), meta_param_lr_dr[:, 0], color=cmap(0))

        # -- arbitrary and bio-inspired terms
        for i in range(meta_param_terms.shape[1]):  # fixme: temp
            plt.plot(range(len(y)), meta_param_terms[:, i], color=cmap(i))

        # for idx, term in enumerate(re.findall(r'(\d+)', test_name)): # fixme: use 'vec' instead of 'test_name'
        #     my_legend.append('Term {}'.format(int(term))) fixme
        #
        #     -- adjust meta param indices for arbitrary terms
        #     if 'arb' in test_name: # fixme: use 'vec' instead of 'test_name'
        #         plt.plot(range(len(y)), meta_param_terms[:, int(term)-1], color=cmap(idx+1))
        #
        #     # -- adjust meta param indices for bio-inspired terms
        #     elif 'bio' in test_name:
        #         plt.plot(range(len(y)), meta_param_terms[:, int(term)-8], color=cmap(idx+1))
        #
        #         # -- adjust meta param index for homeostatic term
        #         if '12' in term:
        #             plt.plot(range(len(y)), meta_param_terms[:, 5], color=cmap(idx+2))
        #             my_legend.append('Term {}.b'.format(int(term)))

        # plt.legend(my_legend)  fixme
        plt.title('Meta parameters')
        # plt.ylim([0, 1])
        plt.savefig(self.res_dir + '/meta_params', bbox_inches='tight')
        plt.close()


        plt.close()

        """


    def meta_angles(self):
        """
            meta angles
        """
        # -- read angles
        y = np.nan_to_num(np.loadtxt(self.res_dir + '/e_ang_meta.txt'))

        for idx in range(1, y.shape[1] - 1):
            # -- compute moving average
            z = self.comp_moving_avg(y[:, idx], self.period)

            # -- plot
            plt.plot(np.array(range(len(z))) + int((self.period - 1) / 2), z, label=r'$\alpha_{}$'.format(idx))

        plt.legend()
        plt.title('Meta Angles')
        plt.savefig(self.res_dir + '/meta_angle', bbox_inches='tight')
        plt.close()

    def meta_loss(self):
        """
            meta loss
        """
        # -- compute moving average
        z = self.comp_moving_avg(np.nan_to_num(np.loadtxt(self.res_dir + '/loss_meta.txt')), self.period)

        # -- plot
        plt.plot(np.array(range(len(z))) + int((self.period - 1) / 2), z)#, label=label, color=self.color)

        plt.title('Meta Loss')
        plt.ylim([0, 5])
        plt.savefig(self.res_dir + '/meta_loss', bbox_inches='tight')
        plt.close()

    def __call__(self, *args, **kwargs):

        self.meta_accuracy()
        self.meta_parameters()
        self.meta_angles()
        self.meta_loss()


def log(data, filename):

    with open(filename, 'a') as f:
        np.savetxt(f, np.array(data), newline=' ', fmt='%0.6f')
        f.writelines('\n')


def normalize_vec(vector):
    """
        normalize input vector.
    """
    return vector / torch.linalg.norm(vector)


def measure_angle(v1, v2):
    """
        Compute angle between two vectors.
    """
    n1 = normalize_vec(v1.squeeze())
    n2 = normalize_vec(v2.squeeze())

    return np.nan_to_num((torch.acos(torch.einsum('i, i -> ', n1, n2)) * 180 / torch.pi).cpu().numpy())


def accuracy(logits, label):

    pred = F.softmax(logits, dim=1).argmax(dim=1)

    return torch.eq(pred, label).sum().item() / len(label)


def meta_stats(logits, params, label, y, Beta, res_dir):

    with torch.no_grad():
        # -- modulator vector stats
        B = dict({k: v for k, v in params.items() if 'fk' in k})
        e = [F.softmax(logits) - F.one_hot(label, num_classes=47)]
        for y_, i in zip(reversed(y), reversed(list(B))):
            e.insert(0, torch.matmul(e[0], B[i]) * (1 - torch.exp(-Beta * y_)))

        # -- orthonormality errors
        W = [v for k, v in params.items() if 'fc' in k]
        E1, E2 = [], []
        activation = [*y, F.softmax(logits, dim=1)]
        for i in range(len(activation)-1):
            E1.append((torch.norm(torch.matmul(activation[i], W[i].T)-torch.matmul(torch.matmul(activation[i+1], W[i]), W[i].T)) ** 2).item())

        log(E1, res_dir + '/E1_meta.txt')

        e_sym = [e[-1]]
        W = dict({k: v for k, v in params.items() if 'fc' in k})
        for y_, i in zip(reversed(y), reversed(list(W))):
            e_sym.insert(0, torch.matmul(e_sym[0], W[i]) * (1 - torch.exp(-Beta * y_)))

        # -- angle b/w modulator vectors e_FA and e_BP
        e_angl = []
        for e_fix_, e_sym_ in zip(e, e_sym):
            e_angl.append(measure_angle(e_fix_.mean(dim=0), e_sym_.mean(dim=0)))

        log(e_angl, res_dir + '/e_ang_meta.txt')

        # -- accuracy
        acc = accuracy(logits, label)

        log([acc], res_dir + '/acc_meta.txt')

    return acc
