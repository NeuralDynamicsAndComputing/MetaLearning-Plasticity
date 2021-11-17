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
                e.insert(0, torch.matmul(e[0], B.clone()) * (1 - torch.exp(-Beta * y)))
                # note: g'(z) = 1 - e^(-Beta*y)

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

class MyOptimizer_(torch.optim.Optimizer): # todo: take matrix multiplications for updates and remove the rest
    """
        Weight update rule.
    """

    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super(MyOptimizer, self).__init__(params, defaults)

    def step(self, loss, y_tuple, logits, feedbacks):
        """
            One step update of the inner-loop.
        :param loss: loss value
        :param y_tuple: input + activations
        :param logits: prediction
        :param feedbacks: feedback matrices
        :return:
        """

        # -- compute error
        e = [torch.autograd.grad(loss, logits, create_graph=True)[0]]
        for y, B in zip(reversed(y_tuple), reversed(feedbacks)):
            e.insert(0, torch.matmul(e[0], B.T) * torch.heaviside(y, torch.tensor(0.0)))

            # todo: check if np.matmul and torch.matmul are the same
            # todo: check if np.heaviside and torch.heaviside are the same

        # -- weight update
        for group in self.param_groups:

            idx = 0
            for i, p in enumerate(group['params']):

                with torch.no_grad():

                    if len(p.shape) == 1:
                        p.add_(e[idx].squeeze(0), alpha=-group['lr'])                       # update biases
                        idx += 1
                    else:
                        p.add_(torch.matmul(e[idx].T, y_tuple[idx]), alpha=-group['lr'])    # update weights

        # with torch.no_grad():
        #     for idx, param in enumerate(self.model.parameters()):
        #         new_param = param - self.lr_innr * grad[idx]
        #         param.copy_(new_param)

        # """ SGD """
        # for group in self.param_groups:
        #     grad = torch.autograd.grad(loss, group['params'], create_graph=True)
        #     for idx, p in enumerate(group['params']):
        #         with torch.no_grad():
        #             if grad[idx] is None:
        #                 continue
        #             p.add_(grad[idx], alpha=-group['lr'])
