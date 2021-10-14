import torch

from torch import optim


class MyOptimizer(optim.Optimizer):

    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super(MyOptimizer, self).__init__(params, defaults)

    def step(self, loss, input, activations, logits, feedbacks):
        """
            Weight update rule.
        :param loss: loss value
        :param y: activations
        :param logits: prediction
        :return:
        """

        n_layers = 4

        # -- compute error
        e = []
        e_L = torch.autograd.grad(loss, logits, create_graph=True)[0]
        e.append(e_L)

        for i in range(n_layers - 2, -1, -1):  # todo: -> for y, B in zip(reversed(activations), reversed(feedbacks)):
            e.insert(0, torch.matmul(e[0], feedbacks[i].T) * torch.heaviside(activations[i], torch.tensor(0.0)))

            # todo: check if np.matmul and torch.matmul are the same
            # todo: check if np.heaviside and torch.heaviside are the same

        

        for group in self.param_groups:

            idx = 0
            for i, p in enumerate(group['params']):

                with torch.no_grad():

                    print(p.shape)

                    print(idx)


                    # todo: -> self.model.get_layers[key].weight.add_(torch.matmul(e[i], y[i].T), alpha=-group['lr'])

                    if len(p.shape) == 1:
                        idx += 1


            quit()


        # -- weight update
        # for group in self.param_groups:
        #
        #     # for i, key in enumerate(self.model.get_layers.keys()):
        #     #     self.model.get_layers[key].weight = self.model.get_layers[key].weight - \
        #     #                                         self.eta * np.matmul(self.e[i], y[i].T)
        #
        #     grad = torch.autograd.grad(loss, group['params'], create_graph=True)
        #
        #     for idx, p in enumerate(group['params']):
        #
        #         with torch.no_grad():
        #             p.add_(grad[idx], alpha=-group['lr'])
        #
        #             # todo: -> self.model.get_layers[key].weight.add_(torch.matmul(e[i], y[i].T), alpha=-group['lr'])


        """ backprop equivalent procedure """
        # for group in self.param_groups:
        #
        #
        #     # -- weight update
        #     for i, key in enumerate(self.model.get_layers.keys()):
        #         self.model.get_layers[key].weight = self.model.get_layers[key].weight - \
        #                                             self.eta * np.matmul(self.e[i], y[i].T)
        #

        """ SGD """
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
