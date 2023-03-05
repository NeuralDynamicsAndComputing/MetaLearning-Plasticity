import torch

from torch.nn import functional as F


def generic_rule(activation, e, params, feedback, Theta, vec, fbk):

    # -- weight update (F_bio)
    i = 0
    for k, p in params.items():
        if 'fc' in k:
            if p.adapt and 'weight' in k:
                p.update = - Theta[0] * torch.matmul(e[i + 1].T, activation[i])

                if '2' in vec:
                    p.update -= Theta[2] * torch.matmul(e[i + 1].T, e[i])
                if '9' in vec:
                    p.update -= Theta[9] * (torch.matmul(activation[i + 1].T, activation[i]) - torch.matmul(
                        torch.matmul(activation[i + 1].T, activation[i + 1]), p))

                params[k] = p + p.update
                params[k].adapt = p.adapt

            i += 1

    if fbk == 'sym':
        # -- feedback update (symmetric)
        feedback_ = dict({k: v for k, v in params.items() if 'fc' in k and 'weight' in k})  # fixme: a vector of k would suffice
        for i, ((k, B), (k_, _)) in enumerate(zip(feedback.items(), feedback_.items())):
            params[k].data = params[k_]
            params[k].adapt = B.adapt


class my_optimizer:
    def __init__(self, update_rule, vec, fbk):
        self.update_rule = update_rule
        self.vec = vec
        self.fbk = fbk

    def __call__(self, params, logits, label, activation, Beta, Theta):

        """
            One step update of the inner-loop (derived formulation).
        :param params: model parameters
        :param logits: unnormalized prediction values
        :param label: target class
        :param activation: vector of activations
        :param Beta: smoothness coefficient for non-linearity
        :param Theta: meta-parameters

        :return:
        """
        # -- error
        feedback = dict({k: v for k, v in params.items() if 'fk' in k})
        e = [F.softmax(logits) - F.one_hot(label, num_classes=47)]
        for y, i in zip(reversed(activation), reversed(list(feedback))):
            e.insert(0, torch.matmul(e[0], feedback[i]) * (1 - torch.exp(-Beta * y)))  # note: g'(z) = 1 - e^(-Beta*y)

        # -- weight update
        self.update_rule([*activation, F.softmax(logits, dim=1)], e, params, feedback, Theta, self.vec, self.fbk)
