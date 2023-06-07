import sys
import torch
from utils import parse_classnames_and_kwargs


def L1_dist_matrix(X, Y):
    return torch.abs(X[:, None] - Y[None, :]).mean(-1)


def efficient_L2_dist_matrix(X, Y):
    """
    Pytorch efficient way of computing distances between all vectors in X and Y, i.e (X[:, None] - Y[None, :])**2
    Get the nearest neighbor index from Y for each X
    :param X:  (n1, d) tensor
    :param Y:  (n2, d) tensor
    Returns a n2 n1 of indices
    """
    dist = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * torch.mm(X, torch.transpose(Y, 0, 1))
    d = X.shape[1]
    dist /= d # normalize by size of vector to make dists independent of the size of d ( use same alpha for all patche-sizes)
    return dist


def batch_dist_matrix(X, Y, b, dist_function):
    """
    For each x find the best index i s.t i = argmin_i(x,y_i)-f_i
    return the value and the argmin
    """
    dists = torch.ones(len(X), len(Y)).to(X.device)
    n_batches = len(X) // b
    for i in range(n_batches):
        s = slice(i * b,(i + 1) * b)
        dists[s] = dist_function(X[s], Y)
    if len(X) % b != 0:
        s = slice(n_batches * b, len(X))
        dists[s] = dist_function(X[s], Y)

    return dists

def batch_NN(X, Y, f, b, dist_function):
    """
    For each x find the best index i s.t i = argmin_i(x,y_i)-f_i
    return the value and the argmin
    """
    NNs = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
    NN_dists = torch.zeros(X.shape[0], device=X.device)
    n_batches = len(X) // b
    for i in range(n_batches):
        s = slice(i * b,(i + 1) * b)
        dists = dist_function(X[s], Y) - f
        NN_dists[s], NNs[s] = dists.min(1)
    if len(X) % b != 0:
        s = slice(n_batches * b, len(X))
        dists = dist_function(X[s], Y) - f
        NN_dists[s], NNs[s] = dists.min(1)

    return NN_dists, NNs


class W1:
    def __init__(self, b=512):
        self.b = b

    def score(self, x, y, f):
        return batch_NN(x, y, f, self.b, L1_dist_matrix)

    def loss(self, x, y):
        return torch.abs(x - y).sum(-1).mean(0)


class W2:
    def __init__(self, b=512):
        self.b = b

    def score(self, x, y, f):
        return batch_NN(x, y, f, self.b, efficient_L2_dist_matrix)

    def loss(self, x, y):
        return ((x - y)**2).sum(-1).mean(0)



def get_loss_function(loss_description):
    loss_name, kwargs = parse_classnames_and_kwargs(loss_description)
    loss = getattr(sys.modules[__name__], loss_name)(**kwargs)
    return loss