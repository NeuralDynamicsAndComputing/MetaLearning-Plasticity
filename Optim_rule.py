import torch

def my_optimizer(params, lr, dr):
    for k, p in params.items():
        if p.adapt:
            p.update = - lr * p.grad
            params[k] = (1 - dr) * p + p.update    # update weight

    return params
