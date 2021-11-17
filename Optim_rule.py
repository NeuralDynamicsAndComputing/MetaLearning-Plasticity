import torch


def my_optimizer(params, loss, logits, activation, Beta, feedback, lr, dr):
    """
        One step update of the inner-loop.
    :param params:
    :param loss: loss value
    :param logits: unnormalized prediction values
    :param activation: vector of activations
    :param Beta: smoothness coefficient for nonlinearity
    :param feedbacks: feedback layers
    :param lr: learning rate variable
    :param dr: damping rate variable
    :return:
    """
    # -- error
    e = [torch.autograd.grad(loss, logits, create_graph=True)[0]]
    for y, layer in zip(reversed(activation), reversed(list(feedback))):
        for k, B in layer.named_parameters():
            if k == 'weight':
                e.insert(0, torch.matmul(e[0], B.clone()) * (1 - torch.exp(-Beta * y)))  # note: g'(z) = 1 - e^(-Beta*y)

    # -- weight update
    i = 0
    for k, p in params.items():
        if p.adapt:
            if k[4:] == 'weight':
                p.update = - lr * torch.matmul(e[i+1].T, activation[i])
                params[k] = (1 - dr) * p + p.update
            elif k[4:] == 'bias':
                p.update = - lr * e[i+1].squeeze(0)
                params[k] = (1 - dr) * p + p.update

                i += 1

    return params
