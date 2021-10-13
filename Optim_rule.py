import torch

from torch import optim


class MyOptimizer(optim.Optimizer):

    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super(MyOptimizer, self).__init__(params, defaults)

    def step(self, loss, y, logits, B):  # todo: -> (self, loss, activations, logits, feedbacks)
        """
            Weight update rule.
        :param loss: loss value
        :param y: activations
        :param logits: prediction
        :return:
        """


        # n_layers = 4
        #
        # # -- compute error
        # e = []
        # e_L = torch.autograd.grad(loss, logits, create_graph=True)[0]
        #
        # e.append(e_L)
        # print(e[0].shape)
        #
        #
        # for i in range(n_layers, 1, -1):
        #     # todo: -> for y, B in zip(activations, feedbacks):
        #     # e.insert(0, torch.matmul(B[i - 2], e[0][0].T) * torch.heaviside(y[i - 2], torch.tensor(0.0)))
        #
        #     print("B : {}".format(B[i - 2].shape))
        #
        #     # print(e[0].shape)
        #
        # quit()
        #
        #
        #     # todo: -> e.insert(0, torch.matmul(B, e[0]) * torch.heaviside(y, 0.0))
        #
        #     # todo: check if np.matmul and torch.matmul are the same
        #     # todo: check if np.heaviside and torch.heaviside are the same
        #
        #
        #
        #
        # for err in e:
        #
        #
        #
        #     print(type(err))
        # quit()

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
        #
        #
        #             p.add_(grad[idx], alpha=-group['lr'])
        #
        #             # todo: -> self.model.get_layers[key].weight.add_(torch.matmul(e[i], y[i].T), alpha=-group['lr'])


        """ backprop equivalent procedure """
        # for group in self.param_groups:
        #
        #     # -- compute error
        #     e = []
        #     e_L = torch.autograd.grad(loss, logits, create_graph=True)
        #     e.append(e_L)
        #
        #     for i in range(n_layers, 1, -1):
        #         self.e.insert(0, torch.matmul(self.B[i - 1], self.e[0]) * torch.heaviside(y[i - 1], 0.0))
        #
        #     # todo: check if np.matmul and torch.matmul are the same
        #     # todo: check if np.heaviside and torch.heaviside are the same
        #     #
        #
        #     # -- weight update
        #     for i, key in enumerate(self.model.get_layers.keys()):
        #         self.model.get_layers[key].weight = self.model.get_layers[key].weight - \
        #                                             self.eta * np.matmul(self.e[i], y[i].T)
        #
        #     quit()
        #
        # def feedback_matrix(self):
        #     feed_mat = {}
        #     for i in range(1, len(self.get_layers)):
        #         feed_mat[i] = self.get_layers[i + 1].weight.T
        #
        #     return feed_mat
        #
        #
        #     # self.e = [y[-1] - y_target]


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
