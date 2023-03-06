import torch

from torch.nn import functional as F


def plasticity_rule(activation, e, params, feedback, Theta, vec, fbk):
    """
        Pool of plasticity rules.

    This function receives the network weights as input and updates them based
    on a meta-learned plasticity rule. It's worth noting that the weights provided
    are a cloned copy of the model's parameters, and that inplace operations can
    be used to update them. Additionally, for model evaluation purposes, use
    'torch.nn.utils._stateless' to replace the module parameters with the cloned
    copy.

    :param activation: (list) model activations,
    :param e: (list) modulatory signals,
    :param params: (dict) model parameters (weights),
    :param feedback: (dict) feedback connections,
    :param Theta: (ParameterList) meta-learned plasticity coefficients,
    :param vec: (list) vector of plasticity rule indices. It determines which
        plasticity rule is applied during the parameter update process.
    :param fbk: (str) the type of feedback matrix used in the model:
        1) 'sym', which indicates that the feedback matrix is symmetric;
        2) 'fix', which indicates that the feedback matrix is a fixed random matrix.)
    :return: None.
    """
    """ update forward weights """
    i = 0
    for k, p in params.items():
        if 'fc' in k:
            if p.adapt and 'weight' in k:
                # -- pseudo-gradient
                p.update = - Theta[0] * torch.matmul(e[i + 1].T, activation[i])

                if '2' in vec:
                    # -- eHebb rule
                    p.update -= Theta[2] * torch.matmul(e[i + 1].T, e[i])
                if '9' in vec:
                    # -- Oja's rule
                    p.update += Theta[9] * (torch.matmul(activation[i + 1].T, activation[i]) - torch.matmul(
                        torch.matmul(activation[i + 1].T, activation[i + 1]), p))

                params[k] = p + p.update
                params[k].adapt = p.adapt

            i += 1

    """ enforce symmetric feedbacks for backprop training """
    if fbk == 'sym':
        # -- feedback update (symmetric)
        feedback_ = dict({k: v for k, v in params.items() if 'fc' in k and 'weight' in k})
        for i, ((k, B), (k_, _)) in enumerate(zip(feedback.items(), feedback_.items())):
            params[k].data = params[k_]
            params[k].adapt = B.adapt


class my_optimizer:
    """
        Adaptation optimizer object.

    The class is responsible for two main operations: computing modulatory
    signals and applying an update rule. The modulatory signals are computed
    based on the current state of the model (activations), and are used to
    adjust the model's parameters. The update rule specifies how these
    adjustments are made.
    """
    def __init__(self, update_rule, vec, fbk):
        """
            Initialize the optimizer

        :param update_rule: (function) weight update function,
        :param vec: (list) vector of plasticity rule indices. It determines which plasticity
            rule is applied during the parameter update process.
        :param fbk: (str) the type of feedback matrix used in the model:
            1) 'sym', which indicates that the feedback matrix is symmetric;
            2) 'fix', which indicates that the feedback matrix is a fixed random matrix.)
        """
        self.update_rule = update_rule
        self.vec = vec
        self.fbk = fbk

    def __call__(self, params, logits, label, activation, Beta, Theta):
        """
            Adaptation loop update.

        The following function is an implementation of one step update of the
        model parameters in the adaptation loop. The function performs the
        following operations:
        1) Computes the modulatory signals using the signal from downstream layers,
            feedback connections, and activations. instead of using pre-activations,
            we use g'(z) = 1 - e^(-Beta*y).
        2) Updates the model parameters using the update function.

        :param params: (dict) model parameters,
        :param logits: (torch.Tensor) unnormalized prediction values,
        :param label: (torch.Tensor) target class,
        :param activation: (tuple) vector of activations,
        :param Beta: (int) smoothness coefficient for non-linearity,
        :param Theta: (ParameterList) plasticity meta-parameters.
        :return: None.
        """
        # -- error
        feedback = dict({k: v for k, v in params.items() if 'fk' in k})
        e = [F.softmax(logits) - F.one_hot(label, num_classes=47)]
        for y, i in zip(reversed(activation), reversed(list(feedback))):
            e.insert(0, torch.matmul(e[0], feedback[i]) * (1 - torch.exp(-Beta * y)))

        # -- weight update
        self.update_rule([*activation, F.softmax(logits, dim=1)], e, params, feedback, Theta, self.vec, self.fbk)
