import torch

from torch.nn import functional as func


def my_optimizer_auto(params, lr, dr):
    """
        One step update of the inner-loop (autograd).
    :param params: model parameters
    :param lr: learning rate variable
    :param dr: decay rate variable
    :return:
    """
    for k, p in params.items():
        if p.adapt:
            p.update = - torch.exp(lr) * p.grad
            params[k] = (1 - torch.exp(dr)) * p + p.update
            params[k].adapt = p.adapt

    return params


def fixed_feedback(activation, e, params, lr, dr):
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


def symmetric_rule(activation, e, params, lr, dr):
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
        self.rule_type = rule_type

        pass

    def __call__(self, params, logits, label, activation, Beta, lr, dr):  # todo: remove loss

        """
            One step update of the inner-loop (derived formulation).
        :param params: model parameters
        :param logits: unnormalized prediction values
        :param label: target class
        :param activation: vector of activations
        :param Beta: smoothness coefficient for non-linearity
        :param lr: learning rate variable
        :param dr: decay rate variable
        :return:
        """
        # -- error
        if self.rule_type == 'symmetric':
            feedback = dict({k: v for k, v in params.items() if 'fc' in k and 'weight' in k})
        elif self.rule_type == 'fixed_feedback':
            feedback = dict({k: v for k, v in params.items() if 'fk' in k})

        # fixme: get total class as an argument
        e = [torch.exp(logits) / torch.sum(torch.exp(logits), dim=1) - func.one_hot(label, num_classes=47)]
        for y, i in zip(reversed(activation), reversed(list(feedback))):
            e.insert(0, torch.matmul(e[0], feedback[i]) * (1 - torch.exp(-Beta * y)))  # note: g'(z) = 1 - e^(-Beta*y)

        # -- update weights
        params = self.update_rule(activation, e, params, lr, dr)

        return params
