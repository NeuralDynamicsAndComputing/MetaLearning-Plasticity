import torch

from torch import optim


class MyOptimizer(optim.Optimizer):

    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super(MyOptimizer, self).__init__(params, defaults)

    def step(self, loss, y, logits):

        """ backprop equivalent procedure """
        # for group in self.param_groups:
        #
        #     e = []
        #     e_L = torch.autograd.grad(loss, logits, create_graph=True)
        #     e.append(e_L)
        #
        #     for i in range(self.n_layers, 1, -1):
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

# def weight_update(self, y, y_target):
#     """
#         Weight update rule.
#     :param y: input, activations, and prediction
#     :param y_target: target label
#     """
#
#     # -- compute error
#     self.e = [y[-1] - y_target]
#     for i in range(self.n_layers, 1, -1):
#         self.e.insert(0, np.matmul(self.B[i - 1], self.e[0]) * np.heaviside(y[i - 1], 0.0))
#
#     # -- weight update
#     for i, key in enumerate(self.model.get_layers.keys()):
#         self.model.get_layers[key].weight = self.model.get_layers[key].weight - \
#                                             self.eta * np.matmul(self.e[i], y[i].T)