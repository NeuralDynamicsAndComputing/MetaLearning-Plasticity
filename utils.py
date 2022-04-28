import os
import re
import torch

import numpy as np
import matplotlib.pyplot as plt


class Plot:
    def __init__(self, n=5000, adpt_idx=[0, 10, 400, 1000, 1100, 1200, 2900, 4900], save_dir='./results/New_Tests/figures/'):
        self.N = n
        self.adpt_idx = adpt_idx
        self.save_dir = save_dir

        for folder in ['Meta acc', 'Meta param', 'Adapt acc', 'Adapt loss', 'Meta ang', 'Meta loss']:
            my_folder = os.path.join(save_dir, folder)
            if not os.path.exists(my_folder):
                os.mkdir(my_folder)

    def meta_accuracy(self, res_dir, test_name):
        """
            meta accuracy
        """

        y = np.nan_to_num(np.loadtxt(res_dir + '/acc_meta.txt'))[:self.N]
        plt.plot(np.array(range(len(y))), y)
        plt.title('Meta Accuracy')
        plt.ylim([0, 1])
        plt.xlim([0, self.N])
        plt.savefig(self.save_dir + 'Meta acc/' + test_name, bbox_inches='tight')
        plt.close()

    def meta_parameters(self, res_dir, test_name):
        """
            meta parameters
        """

        # -- read meta params
        with open(res_dir + '/params.txt', 'r') as f:
            strings = re.findall(r'(-?\d+\.\d+|nan)', f.read())

        y = np.nan_to_num(np.asarray([float(i) for i in strings])).reshape(-1, 21)[:self.N]
        meta_param_lr_dr = y[:, 2:4]
        meta_param_terms = y[:, 4:]

        # -- plot meta params
        my_legend = ['BP']
        cmap = plt.get_cmap("tab10")

        # -- backprop term
        plt.plot(range(len(y)), meta_param_lr_dr[:, 0], color=cmap(0))

        # -- arbitrary and bio-inspired terms
        for idx, term in enumerate(re.findall(r'(\d+)', test_name)):
            my_legend.append('Term {}'.format(int(term)))

            # -- adjust meta param indices for arbitrary terms
            if 'arb' in test_name:
                plt.plot(range(len(y)), meta_param_terms[:, int(term)-1], color=cmap(idx+1))

            # -- adjust meta param indices for bio-inspired terms
            elif 'bio' in test_name:
                plt.plot(range(len(y)), meta_param_terms[:, int(term)-8], color=cmap(idx+1))

                # -- adjust meta param index for homeostatic term
                if '12' in term:
                    plt.plot(range(len(y)), meta_param_terms[:, 5], color=cmap(idx+2))
                    my_legend.append('Term {}.b'.format(int(term)))

        plt.legend(my_legend)
        plt.title('Meta parameters')
        # plt.ylim([0, 1])
        plt.savefig(self.save_dir + 'Meta param/' + test_name, bbox_inches='tight')
        plt.close()

    def adapt_accuracy(self, res_dir, test_name):
        """
            adaptation accuracy
        """

        y = np.nan_to_num(np.loadtxt(res_dir + '/acc.txt'))[:self.N]

        for idx in self.adpt_idx:
            try:
                plt.plot(np.array(range(y.shape[1])), y[idx])
            except IndexError:
                pass

        plt.ylim([0, 1])
        plt.legend(self.adpt_idx)

        plt.title('Adaptation loss')
        plt.savefig(self.save_dir + 'Adapt acc/' + test_name, bbox_inches='tight')
        plt.close()

    def adapt_loss(self, res_dir, test_name):
        """
            adaptation loss
        """

        y = np.nan_to_num(np.loadtxt(res_dir + '/loss.txt'))[:self.N]

        for idx in self.adpt_idx:
            try:
                plt.plot(np.array(range(y.shape[1])), y[idx])
            except IndexError:
                pass

        plt.ylim([0, 5])
        plt.legend(self.adpt_idx)

        plt.title('Adaptation loss')
        plt.savefig(self.save_dir + 'Adapt loss/' + test_name, bbox_inches='tight')
        plt.close()

    def meta_angles(self, res_dir, test_name):
        """
            meta angles
        """

        y = np.nan_to_num(np.loadtxt(res_dir + '/ang_meta.txt'))[:self.N]

        for idx in range(y.shape[1] - 1):
            plt.plot(range(len(y)), y[:, idx])
            plt.legend(['1', '2', '3', '4', '5', '6'])

        plt.title('Meta Angles')
        plt.xlim([0, self.N])
        plt.savefig(self.save_dir + 'Meta ang/' + test_name, bbox_inches='tight')
        plt.close()

    def meta_loss(self, res_dir, test_name):
        """
            meta loss
        """

        y = np.nan_to_num(np.loadtxt(res_dir + '/loss_meta.txt'))[:self.N]
        plt.plot(np.array(range(len(y))), y)
        plt.title('Meta Loss')
        plt.ylim([0, 5])
        plt.xlim([0, self.N])
        plt.savefig(self.save_dir + 'Meta loss/' + test_name, bbox_inches='tight')
        plt.close()

    def __call__(self, *args, **kwargs):
        filename = 'New_Tests/trunk'
        for test_name in [folders for _, folders, _ in os.walk(os.path.join('./results/', filename))][0]:
            for folder in [folders for _, folders, _ in os.walk(os.path.join('./results/', filename, test_name))][0]:
                res_dir = os.path.join('./results/', filename, test_name, folder)

                self.meta_accuracy(res_dir, test_name)
                self.meta_parameters(res_dir, test_name)
                self.adapt_accuracy(res_dir, test_name)
                self.adapt_loss(res_dir, test_name)
                self.meta_angles(res_dir, test_name)
                self.meta_loss(res_dir, test_name)


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
