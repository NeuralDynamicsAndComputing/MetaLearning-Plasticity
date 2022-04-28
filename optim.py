import torch

from torch.nn import functional as F
from utils import normalize_vec, measure_angle


def generic_rule(activation, e, params, feedback, Theta, vec, fbk):
    lr, dr, tre, fur, fiv, six, svn, eit, nin, ten, elv, twl, trt, frt, fif, sxt, svt, etn, ntn = Theta

    # -- weight update
    i = 0
    for k, p in params.items():
        if p.adapt and 'fc' in k:
            if 'weight' in k:
                p.update = - torch.exp(lr) * torch.matmul(e[i + 1].T, activation[i])

                if '1' in vec:
                    p.update -= tre * torch.matmul(activation[i + 1].T, e[i])
                if '3' in vec:
                    p.update -= fiv * torch.matmul(e[i + 1].T, e[i])
                if '8' in vec:
                    p.update -= ten * (torch.matmul(e[i + 1].T, e[i]) - torch.matmul(torch.matmul(e[i + 1].T, e[i + 1]), p))
                if '9' in vec:
                    p.update -= elv * (torch.matmul(e[i + 1].T, e[i]) - torch.matmul(p, torch.matmul(e[i].T, e[i])))
                if '11' in vec:
                    p.update -= trt * (torch.matmul(activation[i + 1].T, e[i]) - torch.matmul(p, torch.matmul(e[i].T, e[i])))

                params[k] = (1 - torch.exp(dr)) * p + p.update
                params[k].adapt = p.adapt

            i += 1

    """# -- feedback update (evolve)
    for i, (k, B) in enumerate(feedback.items()):
        B.update = - torch.exp(lr_fk) * torch.matmul(e[i + 1].T, activation[i])
        params[k] = (1 - torch.exp(dr_fk)) * B + B.update
        params[k].adapt = B.adapt"""

    if fbk == 'sym':
        # -- feedback update (symmetric)
        feedback_ = dict({k: v for k, v in params.items() if 'fc' in k and 'weight' in k})
        for i, ((k, B), (k_, _)) in enumerate(zip(feedback.items(), feedback_.items())):
            params[k].data = params[k_]
            params[k].adapt = B.adapt


def evolve_rule(activation, e, params, feedback, Theta):
    lr_fwd, dr_fwd, lr_fdk, dr_fdk = Theta
    # -- weight update
    i = 0
    for k, p in params.items():
        if p.adapt and 'fc' in k:
            if 'weight' in k:
                p.update = - torch.exp(lr_fwd) * torch.matmul(e[i + 1].T, activation[i])
                params[k] = (1 - torch.exp(dr_fwd)) * p + p.update
                params[k].adapt = p.adapt
            # elif 'bias' in k:
            #     p.update = - torch.exp(lr_fwd) * e[i + 1].squeeze(0)
            #     params[k] = (1 - torch.exp(dr_fwd)) * p + p.update
            #     params[k].adapt = p.adapt

            i += 1

    # -- feedback update
    for i, (k, B) in enumerate(feedback.items()):
        B.update = - torch.exp(lr_fdk) * torch.matmul(e[i + 1].T, activation[i])
        params[k] = (1 - torch.exp(dr_fdk)) * B + B.update
        params[k].adapt = B.adapt

    return params


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
        e = [torch.exp(logits) / torch.sum(torch.exp(logits), dim=1) - F.one_hot(label, num_classes=47)]
        for y, i in zip(reversed(activation), reversed(list(feedback))):
            e.insert(0, torch.matmul(e[0], feedback[i]) * (1 - torch.exp(-Beta * y)))  # note: g'(z) = 1 - e^(-Beta*y)

        # -- compute angles
        with torch.no_grad():
            e_sym_vec = [torch.exp(logits) / torch.sum(torch.exp(logits), dim=1) - F.one_hot(label, num_classes=47)]
            feedback_sym = dict({k: v for k, v in params.items() if 'fc' in k})
            for y, i in zip(reversed(activation), reversed(list(feedback_sym))):
                e_sym_vec.insert(0, torch.matmul(e_sym_vec[0], feedback_sym[i]) * (1 - torch.exp(-Beta * y)))

            angle = []
            for e_fix, e_sym in zip(e, e_sym_vec):
                angle.append(measure_angle(e_fix, e_sym))

        # -- weight update
        self.update_rule([*activation, F.softmax(logits, dim=1)], e, params, feedback, Theta, self.vec, self.fbk)

        return angle

