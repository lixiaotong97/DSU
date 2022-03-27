import torch

def adjust_learning_rate(method, base_lr, iters, max_iters, power):
    if method=='poly':
        lr = base_lr * ((1 - float(iters) / max_iters) ** (power))
    else:
        raise NotImplementedError
    return lr