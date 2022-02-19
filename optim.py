import torch

from torch.nn import functional as F


def generic_rule(activation, e, params, feedback, Theta):
    lr, dr, tre, fur, fiv, six, svn, eit, nin = Theta
    vec = ['1', '2', '3']

    # -- weight update
    i = 0
    for k, p in params.items():
        if p.adapt and 'fc' in k:
            if 'weight' in k:
                p.update = - torch.exp(lr) * torch.matmul(e[i + 1].T, activation[i])

                if '1' in vec:
                    p.update -= tre * torch.matmul(activation[i + 1].T, e[i])
                if '2' in vec:
                    p.update -= fur * torch.matmul(activation[i + 1].T, activation[i])
                if '3' in vec:
                    p.update -= fiv * torch.matmul(e[i + 1].T, e[i])
                if '4' in vec:
                    p.update -= six * activation[i + 1].T.repeat(1, p.shape[1])
                if '5' in vec:
                    p.update -= svn * activation[i].repeat(p.shape[0], 1)
                if '6' in vec:
                    p.update -= eit * e[i + 1].T.repeat(1, p.shape[1])
                if '7' in vec:
                    p.update -= nin * e[i].repeat(p.shape[0], 1)

                params[k] = (1 - torch.exp(dr)) * p + p.update
                params[k].adapt = p.adapt

            i += 1

    """# -- feedback update (evolve)
    for i, (k, B) in enumerate(feedback.items()):
        B.update = - torch.exp(lr_fk) * torch.matmul(e[i + 1].T, activation[i])
        params[k] = (1 - torch.exp(dr_fk)) * B + B.update
        params[k].adapt = B.adapt"""

    """# -- feedback update (symmetric)
    feedback_ = dict({k: v for k, v in params.items() if 'fc' in k and 'weight' in k})
    for i, ((k, B), (k_, _)) in enumerate(zip(feedback.items(), feedback_.items())):
        params[k].data = params[k_]
        params[k].adapt = B.adapt"""


def evolve_rule(activation, e, params, feedback, Theta):
    lr_fwd, dr_fwd, lr_fdk, dr_fdk = Theta
    # -- weight update
    i = 0
    for k, p in params.items():
        if p.adapt and 'fc' in k:
            if 'weight' in k:
                p.update = - torch.exp(lr_fwd) * torch.matmul(e[i + 1].T, activation[i])
                params[k] = (1 - torch.exp(dr_fwd)) * p + p.update
                params[k].adapt = p.adapt
            # elif 'bias' in k:
            #     p.update = - torch.exp(lr_fwd) * e[i + 1].squeeze(0)
            #     params[k] = (1 - torch.exp(dr_fwd)) * p + p.update
            #     params[k].adapt = p.adapt

            i += 1

    # -- feedback update
    for i, (k, B) in enumerate(feedback.items()):
        B.update = - torch.exp(lr_fdk) * torch.matmul(e[i + 1].T, activation[i])
        params[k] = (1 - torch.exp(dr_fdk)) * B + B.update
        params[k].adapt = B.adapt

    return params


def fixed_feedback(activation, e, params, feedback, Theta):
    lr, dr = Theta
    # -- weight update
    i = 0
    for k, p in params.items():
        if p.adapt:
            if 'weight' in k:
                p.update = - torch.exp(lr) * torch.matmul(e[i+1].T, activation[i])
                params[k] = (1 - torch.exp(dr)) * p + p.update
                params[k].adapt = p.adapt
            # elif 'bias' in k:
            #     p.update = - torch.exp(lr) * e[i+1].squeeze(0)
            #     params[k] = (1 - torch.exp(dr)) * p + p.update
            #     params[k].adapt = p.adapt

            i += 1

    return params


def symmetric_rule(activation, e, params, feedback, Theta):
    lr, dr = Theta
    # -- weight update
    i = 0
    for k, p in params.items():
        if p.adapt:
            if 'weight' in k:
                p.update = - torch.exp(lr) * torch.matmul(e[i + 1].T, activation[i])
                params[k] = (1 - torch.exp(dr)) * p + p.update
                params[k].adapt = p.adapt
            # elif 'bias' in k:
            #     p.update = - torch.exp(lr) * e[i + 1].squeeze(0)
            #     params[k] = (1 - torch.exp(dr)) * p + p.update
            #     params[k].adapt = p.adapt

            i += 1

    # -- feedback update
    for i, (k, B) in enumerate(feedback.items()):
        B.update = - torch.exp(lr) * torch.matmul(e[i + 1].T, activation[i])
        params[k].data = (1 - torch.exp(dr)) * B + B.update
        params[k].adapt = B.adapt

    return params


class my_optimizer:
    def __init__(self, update_rule):
        self.update_rule = update_rule

    def __call__(self, params, logits, label, activation, Beta, Theta):

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
        e = [torch.exp(logits) / torch.sum(torch.exp(logits), dim=1) - F.one_hot(label, num_classes=47)]
        for y, i in zip(reversed(activation), reversed(list(feedback))):
            e.insert(0, torch.matmul(e[0], feedback[i]) * (1 - torch.exp(-Beta * y)))  # note: g'(z) = 1 - e^(-Beta*y)

        # -- weight update
        self.update_rule([*activation, F.softmax(logits, dim=1)], e, params, feedback, Theta)
