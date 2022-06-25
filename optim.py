import torch

from torch.nn import functional as F
from utils import normalize_vec, measure_angle


def generic_rule(activation, e, params, feedback, Theta, vec, fbk):
    lr, tre, fiv, nin, elv, trt, frt, etn, ntn = Theta

    # -- weight update
    i = 0
    for k, p in params.items():
        if 'fc' in k:
            if p.adapt and 'weight' in k:
                p.update = - lr * torch.matmul(e[i + 1].T, activation[i])

                if '1' in vec:
                    p.update -= tre * torch.matmul(activation[i + 1].T, e[i])
                if '3' in vec:
                    p.update -= fiv * torch.matmul(e[i + 1].T, e[i])
                if '7' in vec:
                    p.update -= nin * e[i].repeat(p.shape[0], 1)
                if '9' in vec:
                    p.update -= elv * (torch.matmul(e[i + 1].T, e[i]) - torch.matmul(p, torch.matmul(e[i].T, e[i])))
                if '11' in vec:
                    p.update -= trt * (torch.matmul(activation[i + 1].T, e[i]) - torch.matmul(p, torch.matmul(e[i].T, e[i])))
                if '12' in vec:
                    p.update -= frt * (torch.matmul(activation[i + 1].T, activation[i]) - torch.matmul(torch.matmul(activation[i + 1].T, activation[i + 1]), p))
                if '16' in vec:
                    p.update -= etn * torch.matmul(e[i + 1].T, (e[i] - ntn))  # todo: try the other way around

                params[k] = p + p.update
                params[k].adapt = p.adapt

            i += 1

    """# -- feedback update (evolve)
    for i, (k, B) in enumerate(feedback.items()):
        B.update = - torch.exp(lr_fk) * torch.matmul(e[i + 1].T, activation[i])
        params[k] = (1 - torch.exp(dr_fk)) * B + B.update
        params[k].adapt = B.adapt"""

    if fbk == 'sym':
        # -- feedback update (symmetric)
        feedback_ = dict({k: v for k, v in params.items() if 'fc' in k and 'weight' in k})  # fixme: a vector of k would suffice
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
        e = [F.softmax(logits) - F.one_hot(label, num_classes=47)]
        for y, i in zip(reversed(activation), reversed(list(feedback))):
            e.insert(0, torch.matmul(e[0], feedback[i]) * (1 - torch.exp(-Beta * y)))  # note: g'(z) = 1 - e^(-Beta*y)

        # -- compute angles
        with torch.no_grad():
            e_sym_vec = [F.softmax(logits) - F.one_hot(label, num_classes=47)]
            feedback_sym = dict({k: v for k, v in params.items() if 'fc' in k})
            for y, i in zip(reversed(activation), reversed(list(feedback_sym))):
                e_sym_vec.insert(0, torch.matmul(e_sym_vec[0], feedback_sym[i]) * (1 - torch.exp(-Beta * y)))

            # -- angle b/w modulator vectors e_FA and e_BP
            angle = []
            for e_fix, e_sym in zip(e, e_sym_vec):
                angle.append(measure_angle(e_fix, e_sym))

            # -- compute angle between update direction and gradient
            lr, tre, fiv, nin, elv, trt, frt, etn, ntn = Theta

            y = [*activation, F.softmax(logits, dim=1)]
            angle_grad, my_grad_, sym_grad_, angle_grad_vec = [], [], [], []
            for l in range(1, len(e_sym_vec)):
                sym_grad = - torch.matmul(e_sym_vec[l].T, activation[l - 1])        # symmetric backprop

                if '12' in self.vec:
                    my_grad = - lr * torch.matmul(e[l].T, y[l - 1]) - frt * (torch.matmul(y[l].T, y[l - 1])
                                - torch.matmul(torch.matmul(y[l].T, y[l]), feedback_sym['fc{}.weight'.format(l)]))
                    # oja + fixed backprop
                elif '16' in self.vec:
                    my_grad = - lr * torch.matmul(e[l].T, activation[l - 1]) \
                              - etn * torch.matmul(e[l].T, (e[l - 1] - ntn))        # homeostatic + fixed backprop
                else:
                    my_grad = - lr * torch.matmul(e[l].T, activation[l - 1])

                angle_grad.append(measure_angle(sym_grad.ravel(), my_grad.ravel()))
                my_grad_.append(my_grad.ravel())
                sym_grad_.append(sym_grad.ravel())

            angle_grad_vec.append(measure_angle(torch.concat(sym_grad_), torch.concat(my_grad_)))

            # -- mean and sd for modulator vectors
            e_mean, e_std, e_norm = [], [], []
            for e_fix in e:
                e_mean.append(e_fix.mean().item())
                e_std.append(e_fix.std().item())
                e_norm.append(torch.norm(e_fix))

            # - angle b/w W and B and norm, mean, and SD for W
            angle_WB, norm_W, W_mean, W_std = [], [], [], []
            for i, i_sym in zip(feedback, feedback_sym):
                angle_WB.append(measure_angle(feedback[i].ravel(), feedback_sym[i_sym].ravel()))
                norm_W.append(torch.norm(feedback_sym[i_sym]))
                W_std.append(feedback_sym[i_sym].std().item())
                W_mean.append(feedback_sym[i_sym].mean().item())

            # -- mean and SD for activations
            y_mean, y_std, y_norm = [], [], []
            for y in [*activation, F.softmax(logits, dim=1)]:
                y_mean.append(y.mean().item())
                y_std.append(y.std().item())
                y_norm.append(torch.norm(y))

        # -- weight update
        self.update_rule([*activation, F.softmax(logits, dim=1)], e, params, feedback, Theta, self.vec, self.fbk)

        return angle, angle_grad, angle_grad_vec, e_mean, e_std, e_norm, angle_WB, norm_W, W_mean, W_std, y_mean, y_std, y_norm
