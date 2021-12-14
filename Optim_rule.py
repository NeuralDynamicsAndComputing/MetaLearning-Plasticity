import torch

from torch.nn import functional as func


def my_optimizer(params, logits, label, activation, Beta, lr, dr, lr_fdk, dr_fdk):
    """
        One step update of the inner-loop.
    :param params:
    :param logits: unnormalized prediction values
    :param label: target class
    :param activation: vector of activations
    :param Beta: smoothness coefficient for non-linearity
    :param lr: learning rate variable for feedforward weight
    :param dr: decay rate variable for feedforward weight
    :param lr_fdk: learning rate variable for feedback connection
    :param dr_fdk: decay rate variable for feedback connection
    :return:
    """
    # -- error
    e = [torch.exp(logits)/torch.sum(torch.exp(logits), dim=1) - func.one_hot(label, num_classes=47)]  # fixme: get total class as an argument
    feedback = dict({k: v for k, v in params.items() if 'fk' in k})
    for y, i in zip(reversed(activation), reversed(list(feedback))):
        e.insert(0, torch.matmul(e[0], feedback[i]) * (1 - torch.exp(-Beta * y)))  # note: g'(z) = 1 - e^(-Beta*y)

    # -- weight update
    i = 0
    for k, p in params.items():
        if p.adapt and 'fc' in k:
            if k[4:] == 'weight':
                p.update = - torch.exp(lr) * torch.matmul(e[i+1].T, activation[i])
                params[k] = (1 - torch.exp(dr)) * p + p.update
                params[k].adapt = p.adapt
            elif k[4:] == 'bias':
                p.update = - torch.exp(lr) * e[i+1].squeeze(0)
                params[k] = (1 - torch.exp(dr)) * p + p.update
                params[k].adapt = p.adapt

                i += 1

    # -- feedback update
    for i, (k, B) in enumerate(feedback.items()):
        B.update = - torch.exp(lr_fdk) * torch.matmul(e[i+1].T, activation[i])
        tmp = params[k].adapt  # todo: needs to be fixed
        params[k] = (1 - torch.exp(dr_fdk)) * B + B.update
        params[k].adapt = tmp
        
    return params
