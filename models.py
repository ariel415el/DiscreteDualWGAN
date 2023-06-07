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



# class PatchWrapper(nn.Module):
#     def __init__(self, G):
#         super(PatchWrapper, self).__init__()
#         self.G = G
#
#     def forward(self, x):
#