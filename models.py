import torch
from torch import nn


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.zero_()

def get_generator(nz, n_hidden, output_dim):
    G = nn.Sequential(
        nn.Linear(nz, n_hidden),
        nn.ReLU(True),
        nn.Linear(n_hidden, n_hidden),
        nn.ReLU(True),
        nn.Linear(n_hidden, n_hidden),
        nn.ReLU(True),
        nn.Linear(n_hidden, output_dim),
        nn.Tanh()
    )

    initialize_weights(G)

    return G

class PixelGenerator(nn.Module):
    def __init__(self, z_dim, output_dim, n=64, init_mode='noise'):
        super(PixelGenerator, self).__init__()
        n = int(n)
        if init_mode == "noise":
            images = torch.randn(n, 3*output_dim**2) * 0.5
        elif init_mode == "ones":
            images = torch.ones(n, 3*output_dim**2)
        elif init_mode == "zeros":
            images = torch.ones(n, 3, 3*output_dim**2)
        else:
            raise ValueError("Bad init mode")
        self.images = nn.Parameter(images, requires_grad=True)
        # self.clip()

    def forward(self, input):
        return torch.tanh(self.images)

# class PatchWrapper(nn.Module):
#     def __init__(self, G):
#         super(PatchWrapper, self).__init__()
#         self.G = G
#
#     def forward(self, x):
#