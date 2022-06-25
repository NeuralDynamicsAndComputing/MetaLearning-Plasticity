import os
import re
import torch

import numpy as np
import matplotlib.pyplot as plt

from torch.nn import functional as F


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

    def adapt_accuracy(self, filename='/acc.txt', savename='/adapt_accuracy', idx_plot=None):
        """
            adaptation accuracy
        """

        y = np.nan_to_num(np.loadtxt(self.res_dir + filename, ndmin=2))

        if idx_plot is not None:
            self.adpt_idx = idx_plot

        for idx in self.adpt_idx:
            try:
                plt.plot(np.array(range(y.shape[1])), y[idx])
            except IndexError:
                pass

        plt.ylim([0, 1])
        plt.legend(self.adpt_idx)

        plt.title('Adaptation loss')
        plt.savefig(self.res_dir + savename, bbox_inches='tight')
        plt.close()

    def adapt_loss(self, filename='/loss.txt', savename='/adapt_loss', idx_plot=None):
        """
            adaptation loss
        """
        y = np.nan_to_num(np.loadtxt(self.res_dir + filename, ndmin=2))

        if idx_plot is not None:
            self.adpt_idx = idx_plot

        for idx in self.adpt_idx:
            try:
                plt.plot(np.array(range(y.shape[1])), y[idx])
            except IndexError:
                pass

        plt.ylim([0, 5])
        plt.legend(self.adpt_idx)

        plt.title('Adaptation loss')
        plt.savefig(self.res_dir + savename, bbox_inches='tight')
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


def accuracy(logits, label):

    pred = F.softmax(logits, dim=1).argmax(dim=1)

    return torch.eq(pred, label).sum().item() / len(label)


def meta_stats(logits, params, label, y, Beta, res_dir):

    with torch.no_grad():
        # -- activation stats
        y_norm, y_mean, y_std = [], [], []
        for y_ in [*y, F.softmax(logits, dim=1)]:
            y_norm.append(y_.norm(dim=1).mean().item())
            y_mean.append(y_.mean().item())
            y_std.append(y_.std(dim=1).mean().item())

        # -- modulator vector stats
        feedback = dict({k: v for k, v in params.items() if 'fk' in k})
        e = [F.softmax(logits) - F.one_hot(label, num_classes=47)]
        for y_, i in zip(reversed(y), reversed(list(feedback))):
            e.insert(0, torch.matmul(e[0], feedback[i]) * (1 - torch.exp(-Beta * y_)))

        e_norm, e_mean, e_std = [], [], []
        for e_ in e:
            e_norm.append(e_.norm(dim=1).mean().item())
            e_mean.append(e_.mean().item())
            e_std.append(e_.std(dim=1).mean().item())

        e_sym = [e[-1]]
        feedback_sym = dict({k: v for k, v in params.items() if 'fc' in k})
        for y_, i in zip(reversed(y), reversed(list(feedback_sym))):
            e_sym.insert(0, torch.matmul(e_sym[0], feedback_sym[i]) * (1 - torch.exp(-Beta * y_)))

        # -- angle b/w modulator vectors e_FA and e_BP
        e_angl = []
        for e_fix_, e_sym_ in zip(e, e_sym):
            e_angl.append(measure_angle(e_fix_.mean(dim=0), e_sym_.mean(dim=0)))

        # -- weight stats
        w_norm, w_mean, w_std = [], [], []
        for w in [v for k, v in params.items() if 'fc' in k]:
            w_norm.append(w.norm().item())
            w_mean.append(w.mean().item())
            w_std.append(w.std().item())

        # -- accuracy
        acc = accuracy(logits, label)

        log([acc], res_dir + '/acc_meta.txt')
        log(w_norm, res_dir + '/norm_W_meta.txt')
        log(w_mean, res_dir + '/W_mean_meta.txt')
        log(w_std, res_dir + '/W_std_meta.txt')
        log(e_norm, res_dir + '/e_norm_meta.txt')
        log(e_mean, res_dir + '/e_mean_meta.txt')
        log(e_std, res_dir + '/e_std_meta.txt')
        log(e_angl, res_dir + '/ang_meta.txt')
        log(y_norm, res_dir + '/y_norm_meta.txt')
        log(y_mean, res_dir + '/y_mean_meta.txt')
        log(y_std, res_dir + '/y_std_meta.txt')


        """ 
        if False:
            error = {'0': [], '1': [], '2': [], '3': [], '4': []}
            SVD_max = {'0': [], '1': [], '2': [], '3': [], '4': []}
            SVD_min = {'0': [], '1': [], '2': [], '3': [], '4': []}
            SVD_mean = {'0': [], '1': [], '2': [], '3': [], '4': []}
            SVD_std = {'0': [], '1': [], '2': [], '3': [], '4': []}
    
        if False:  # todo: take func to utils
            W = dict({k: v for k, v in params.items() if 'fc' in k})
            for i, (k, w) in enumerate(W.items()):
                if False:
                    err = torch.norm(torch.matmul(y[i], w.T) -
                                     torch.matmul(torch.matmul(self.model.sopl(torch.matmul(y[i], w.T)), w), w.T)) ** 2
                    log([err], self.res_dir + '/err_{}_meta.txt'.format(k[2]))
                if True:
                    err = torch.mean(torch.norm(y[i] - torch.matmul(y[i], torch.matmul(w.T, w)), dim=1) ** 2).item()
                    log([err], self.res_dir + '/err_{}_meta.txt'.format(k[2]))
    
                s = np.linalg.svd(w.cpu().detach().numpy(), compute_uv=False)
    
                log([np.max(s)], self.res_dir + '/svd_max_{}_meta.txt'.format(k[2]))
                log([np.min(s)], self.res_dir + '/svd_min_{}_meta.txt'.format(k[2]))
                log([np.mean(s)], self.res_dir + '/svd_mean_{}_meta.txt'.format(k[2]))
                log([np.std(s)], self.res_dir + '/svd_std_{}_meta.txt'.format(k[2]))
    
        if False:
            for k, err in error.items():
                log(err, self.res_dir + '/err_{}.txt'.format(k))
    
            for k, s in SVD_max.items():
                log(s, self.res_dir + '/svd_max_{}.txt'.format(k))
    
            for k, s in SVD_min.items():
                log(s, self.res_dir + '/svd_min_{}.txt'.format(k))
    
            for k, s in SVD_mean.items():
                log(s, self.res_dir + '/svd_mean_{}.txt'.format(k))
    
            for k, s in SVD_std.items():
                log(s, self.res_dir + '/svd_std_{}.txt'.format(k))
    
    
    
        # log([angle_grad_vec], self.res_dir + '/angle_grad_vec.txt')
        # log(angle_grad, self.res_dir + '/ang_grad_meta.txt')
        # log(angles, self.res_dir + '/ang.txt')
        # log(angles_grad, self.res_dir + '/ang_grad.txt')
    
        if False:  # todo: take to utils
            if eps % 10 == 0:
    
                W_1 = params['fc1.weight'].ravel().detach().cpu().numpy()
                W_2 = params['fc2.weight'].ravel().detach().cpu().numpy()
                W_3 = params['fc3.weight'].ravel().detach().cpu().numpy()
                W_4 = params['fc4.weight'].ravel().detach().cpu().numpy()
                W_5 = params['fc5.weight'].ravel().detach().cpu().numpy()
    
                W = np.concatenate((W_1, W_2, W_3, W_4, W_5))
    
                n_bins = 100
                title = ['W1', 'W2', 'W3', 'W4', 'W5', 'W']
                for idx_t, w in enumerate([W_1, W_2, W_3, W_4, W_5, W]):
    
                    weights = np.ones_like(w) / float(len(w))
                    prob, bins, _ = plt.hist(w, n_bins, range=(-1, 1), density=False, histtype='step', color='red', weights=weights)
                    plt.close()
                    log(prob, self.res_dir + '/prob{}.txt'.format(title[idx_t]))
                    log(bins, self.res_dir + '/bins{}.txt'.format(title[idx_t]))
        """

        """
        # -- compute angles   # todo: remove
        if False:
            with torch.no_grad():
                e_sym_vec = [F.softmax(logits) - F.one_hot(label, num_classes=47)]
                feedback_sym = dict({k: v for k, v in params.items() if 'fc' in k})
                for y, i in zip(reversed(activation), reversed(list(feedback_sym))):
                    e_sym_vec.insert(0, torch.matmul(e_sym_vec[0], feedback_sym[i]) * (1 - torch.exp(-Beta * y)))
    
                # -- angle b/w modulator vectors e_FA and e_BP  # todo: remove
                angle = []
                for e_fix, e_sym in zip(e, e_sym_vec):
                    angle.append(measure_angle(e_fix, e_sym))
    
                # -- compute angle between update direction and gradient  # todo: remove
                lr, tre, fiv, nin, elv, trt, frt, etn, ntn = Theta
    
                y = [*activation, F.softmax(logits, dim=1)]
                angle_grad, my_grad_, sym_grad_, angle_grad_vec = [], [], [], []
                for l in range(1, len(e_sym_vec)):
                    sym_grad = - torch.matmul(e_sym_vec[l].T, activation[l - 1])        # symmetric backprop
    
                    if '12' in self.vec:
                        my_grad = - lr * torch.matmul(e[l].T, y[l - 1]) - frt * (torch.matmul(y[l].T, y[l - 1])
                                    - torch.matmul(torch.matmul(y[l].T, y[l]), feedback_sym['fc{}.weight'.format(l)]))
                        # oja + fixed backprop
                    elif '16' in self.vec:
                        my_grad = - lr * torch.matmul(e[l].T, activation[l - 1]) \
                                  - etn * torch.matmul(e[l].T, (e[l - 1] - ntn))        # homeostatic + fixed backprop
                    else:
                        my_grad = - lr * torch.matmul(e[l].T, activation[l - 1])
    
                    angle_grad.append(measure_angle(sym_grad.ravel(), my_grad.ravel()))
                    my_grad_.append(my_grad.ravel())
                    sym_grad_.append(sym_grad.ravel())
    
                angle_grad_vec.append(measure_angle(torch.concat(sym_grad_), torch.concat(my_grad_)))
        """

    return acc
