import torch

from torch.nn import functional as func


def my_optimizer(params, logits, label, activation, Beta, lr, dr):  # todo: remove loss
    """
        One step update of the inner-loop.
    :param params: model parameters
    :param logits: activations z_L at the last layer
    :param label: target class
    :param activation: network activations z_l
    :param Beta: smoothness coefficient for the softplus non-linearity
    :param lr: learning rate variable
    :param dr: decay rate variable
    :return: updated adaptation parameters
    """
    # -- error
    e = [torch.exp(logits)/torch.sum(torch.exp(logits), dim=1) - func.one_hot(label, num_classes=47)]  # fixme: get total class as an argument
    feedback = dict({k: v for k, v in params.items() if 'fk' in k})  # todo: consider adding bias
    for y, i in zip(reversed(activation), reversed(list(feedback))):
        e.insert(0, torch.matmul(e[0], feedback[i]) * (1 - torch.exp(-Beta * y)))  # note: g'(z) = 1 - e^(-Beta*y)

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
