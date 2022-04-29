import os
import re
import torch

import numpy as np
import matplotlib.pyplot as plt


class Plot:
    def __init__(self, res_dir, adpt_idx=[0, 10, 400, 1000, 1100, 1200, 2900, 4900]):
        self.adpt_idx = adpt_idx
        self.res_dir = res_dir

    def meta_accuracy(self):
        """
            meta accuracy
        """

        y = np.nan_to_num(np.loadtxt(self.res_dir + '/acc_meta.txt'))
        plt.plot(np.array(range(len(y))), y)
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

    def adapt_accuracy(self):
        """
            adaptation accuracy
        """

        y = np.nan_to_num(np.loadtxt(self.res_dir + '/acc.txt'))

        for idx in self.adpt_idx:
            try:
                plt.plot(np.array(range(y.shape[1])), y[idx])
            except IndexError:
                pass

        plt.ylim([0, 1])
        plt.legend(self.adpt_idx)

        plt.title('Adaptation loss')
        plt.savefig(self.res_dir + '/adapt_accuracy', bbox_inches='tight')
        plt.close()

    def adapt_loss(self):
        """
            adaptation loss
        """

        y = np.nan_to_num(np.loadtxt(self.res_dir + '/loss.txt'))

        for idx in self.adpt_idx:
            try:
                plt.plot(np.array(range(y.shape[1])), y[idx])
            except IndexError:
                pass

        plt.ylim([0, 5])
        plt.legend(self.adpt_idx)

        plt.title('Adaptation loss')
        plt.savefig(self.res_dir + '/adapt_loss', bbox_inches='tight')
        plt.close()

    def meta_angles(self):
        """
            meta angles
        """

        y = np.nan_to_num(np.loadtxt(self.res_dir + '/ang_meta.txt'))

        for idx in range(y.shape[1] - 1):
            plt.plot(range(len(y)), y[:, idx])
            plt.legend(['1', '2', '3', '4', '5', '6'])

        plt.title('Meta Angles')
        plt.savefig(self.res_dir + '/meta_angle', bbox_inches='tight')
        plt.close()

    def meta_loss(self):
        """
            meta loss
        """

        y = np.nan_to_num(np.loadtxt(self.res_dir + '/loss_meta.txt'))
        plt.plot(np.array(range(len(y))), y)
        plt.title('Meta Loss')
        plt.ylim([0, 5])
        plt.savefig(self.res_dir + '/meta_loss', bbox_inches='tight')
        plt.close()

    def __call__(self, *args, **kwargs):

        self.meta_accuracy()
        self.meta_parameters()
        self.adapt_accuracy()
        self.adapt_loss()
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
