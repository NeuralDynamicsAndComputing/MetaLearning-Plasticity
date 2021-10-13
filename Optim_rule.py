import torch

from torch import optim


class MyOptimizer(optim.Optimizer):

    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super(MyOptimizer, self).__init__(params, defaults)

    def step(self, loss):

        for group in self.param_groups:

            grad = torch.autograd.grad(loss, group['params'], create_graph=True)

            for idx, p in enumerate(group['params']):

                if False:
                    p.grad = grad[idx]

                    with torch.no_grad():

                        if p.grad is None:
                            continue
                        d_p = p.grad

                        p.add_(d_p, alpha=-group['lr'])
                else:
                    with torch.no_grad():

                        if grad[idx] is None:
                            continue

                        p.add_(grad[idx], alpha=-group['lr'])