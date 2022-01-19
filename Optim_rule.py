import torch

from torch.nn import functional as func


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
            elif 'bias' in k:
                p.update = - torch.exp(lr_fwd) * e[i + 1].squeeze(0)
                params[k] = (1 - torch.exp(dr_fwd)) * p + p.update
                params[k].adapt = p.adapt

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
            if k[4:] == 'weight':
                p.update = - torch.exp(lr) * torch.matmul(e[i+1].T, activation[i])
                params[k] = (1 - torch.exp(dr)) * p + p.update
                params[k].adapt = p.adapt
            elif k[4:] == 'bias':
                p.update = - torch.exp(lr) * e[i+1].squeeze(0)
                params[k] = (1 - torch.exp(dr)) * p + p.update
                params[k].adapt = p.adapt

                i += 1

    return params


def symmetric_rule(activation, e, params, feedback, Theta):
    lr, dr = Theta
    # -- weight update
    i = 0
    for k, p in params.items():
        if p.adapt:
            if k[4:] == 'weight':
                p.update = - torch.exp(lr) * torch.matmul(e[i + 1].T, activation[i])
                params[k] = (1 - torch.exp(dr)) * p + p.update
                params[k].adapt = p.adapt
            elif k[4:] == 'bias':
                p.update = - torch.exp(lr) * e[i + 1].squeeze(0)
                params[k] = (1 - torch.exp(dr)) * p + p.update
                params[k].adapt = p.adapt

                i += 1

    return params


class my_optimizer:
    def __init__(self, update_rule, rule_type):
        self.update_rule = update_rule
        self.rule_type = rule_type  # todo: remove this

        self.sym, self.fix, self.evl = False, False, True  # todo: take this to args

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
        if self.sym:
            feedback = dict({k: v for k, v in params.items() if 'fc' in k and 'weight' in k})
        elif self.fix:
            feedback = dict({k: v for k, v in params.items() if 'fk' in k})
        elif self.evl:
            feedback = dict({k: v for k, v in params.items() if 'fk' in k})

        # fixme: get total class as an argument
        e = [torch.exp(logits) / torch.sum(torch.exp(logits), dim=1) - func.one_hot(label, num_classes=47)]
        for y, i in zip(reversed(activation), reversed(list(feedback))):
            e.insert(0, torch.matmul(e[0], feedback[i]) * (1 - torch.exp(-Beta * y)))  # note: g'(z) = 1 - e^(-Beta*y)

        # -- update weights
        params = self.update_rule(activation, e, params, feedback, Theta)

        return params
