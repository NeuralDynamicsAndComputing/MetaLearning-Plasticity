import torch
from torch import nn

def my_optimizer(params, loss, logits, activation, Beta, lr, dr, lr_fdb, dr_fdb):
    """
        One step update of the inner-loop.
    :param params:
    :param loss: loss value
    :param logits: unnormalized prediction values
    :param activation: vector of activations
    :param Beta: smoothness coefficient for nonlinearity
    :param lr: learning rate variable
    :param dr: damping rate variable
    :return:
    """
    # -- error
    e = [torch.autograd.grad(loss, logits, create_graph=True)[0]]
    # todo: make a list of feedback keys only?
    feedback = dict({k: v for k, v in params.items() if 'fk' in k})  # todo: add bias later
    for y, i in zip(reversed(activation), reversed(list(feedback))):  # todo: transpose B
        e.insert(0, torch.matmul(e[0], feedback[i]) * (1 - torch.exp(-Beta * y)))  # note: g'(z) = 1 - e^(-Beta*y) # todo: add bias later

    # -- weight update
    i = 0
    for k, p in params.items():
        if p.adapt and 'fc' in k:
            if k[4:] == 'weight':
                p.update = - lr * torch.matmul(e[i+1].T, activation[i])
                params[k] = (1 - dr) * p + p.update
            elif k[4:] == 'bias':
                p.update = - lr * e[i+1].squeeze(0)
                params[k] = (1 - dr) * p + p.update
                i += 1

    # -- feedback update
    for i, (key, B) in enumerate(feedback.items()):
        B.update = - lr_fdb * torch.matmul(e[i+1].T, activation[i])
        params[key] = (1 - dr_fdb) * B + B.update

    return params
