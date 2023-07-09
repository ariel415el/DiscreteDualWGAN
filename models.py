import torch
from torch import nn

class PixelGenerator(nn.Module):
    def __init__(self, z_dim, output_dim, n=64):
        super(PixelGenerator, self).__init__()
        self.images = nn.Parameter(torch.randn(n, 3*output_dim**2), requires_grad=True)

    def forward(self, input):
        return torch.tanh(self.images)


class FCGenerator(nn.Module):
    def __init__(self, z_dim, output_dim, n_hidden=64):
        super(FCGenerator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(z_dim, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, output_dim),
            nn.Tanh()
        )

    def forward(self, input):
        return self.layers(input)



class FCDiscriminator(nn.Module):
    def __init__(self, input_dim, n_hidden=64):
        super(FCDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3 * input_dim**2, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, input):
        # return self.layers(input.reshape(input.shape[0], -1)).reshape(-1)
        return self.layers(input).reshape(-1)


def conv_block(c_in, c_out, k_size, stride, pad, normalize='in', transpose=False):
    module = []

    conv_type = nn.ConvTranspose2d if transpose else nn.Conv2d
    module.append(conv_type(c_in, c_out, k_size, stride, pad, bias=normalize is None))

    if normalize == "bn":
        module.append(nn.BatchNorm2d(c_out))
    elif normalize == "in":
        module.append(nn.InstanceNorm2d(c_out))

    module.append(nn.ReLU(True))
    return nn.Sequential(*module)



class DCGenerator(nn.Module):
    def __init__(self, z_dim, nf='64',  normalize='none', **kwargs):
        super(DCGenerator, self).__init__()
        channels = 3
        nf = int(nf)
        normalize = str(normalize)
        layer_depths = [z_dim, nf*8, nf*4, nf*2, nf]
        kernel_dim = [4, 4, 4, 4, 4]
        strides = [1, 2, 2, 2, 2]
        padding = [0, 1, 1, 1, 1]

        layers = []
        for i in range(len(layer_depths) - 1):
            layers.append(
                conv_block(layer_depths[i], layer_depths[i + 1], kernel_dim[i], strides[i], padding[i], normalize=normalize, transpose=True)
            )
        layers += [
            nn.ConvTranspose2d(layer_depths[-1], channels, kernel_dim[-1], strides[-1], padding[-1]),
            nn.Tanh()
        ]
        self.network = nn.Sequential(*layers)

    def forward(self, input):
        input = input.view(input.size(0), input.size(1), 1, 1)
        output = self.network(input).reshape(input.size(0), -1)
        return output

class DCDiscriminator(nn.Module):
    """ DC-discriminator receptive field by layer (4, 10, 22, 46, 94)"""

    def __init__(self, input_dim=64, nf='64', normalize='none', num_outputs=1, **kwargs):
        super(DCDiscriminator, self).__init__()
        channels = 3
        nf = int(nf)
        normalize = str(normalize)
        self.input_dim = int(input_dim)
        layer_depth = [channels, nf, nf * 2, nf * 4, nf * 8]

        layers = []
        for i in range(len(layer_depth) - 1):
            normalize_layer = normalize if i > 0 else 'none'  # bn is not good for RGB values
            layers.append(
                conv_block(layer_depth[i], layer_depth[i + 1], 4, 2, 1, normalize=normalize_layer, transpose=False)
            )
        self.convs = nn.Sequential(*layers)
        self.classifier = nn.Linear(layer_depth[-1] * 4 ** 2, num_outputs)
        self.num_outputs = num_outputs

    def features(self, img):
        return self.convs(img)

    def forward(self, img):
        b = img.size(0)
        features = self.convs(img.reshape(b, 3, self.input_dim, self.input_dim))

        features = features.reshape(b, -1)

        output = self.classifier(features)
        if self.num_outputs == 1:
            output = output.view(len(img))
        else:
            output = output.view(len(img), self.num_outputs)

        return output


class PatchGAN(nn.Module):
    def __init__(self, input_dim, depth=3, nf=64, normalize='in', k=3, pad=0, **kwargs):
        super(PatchGAN, self).__init__()
        channels=3
        depth = int(depth)
        nf = int(nf)
        k = int(k)
        normalize = str(normalize)
        self.input_dim = input_dim

        layers = [conv_block(channels, nf, k, 2, pad, normalize='none', transpose=False)]  # bn is not good for RGB values
        for i in range(depth - 1):
            layers.append(
                conv_block(nf * 2**i, nf * 2**(i+1), k, 2, pad, normalize=normalize, transpose=False)
            )
        self.convs = nn.Sequential(*layers)
        self.classifier = nn.Linear(nf * 2**(depth-1), 1, bias=False)

    def features(self, img):
        return self.convs(img)

    def forward(self, img):
        b = img.size(0)
        features = self.convs(img.reshape(b, 3, self.input_dim, self.input_dim))
        features = torch.mean(features, dim=(2, 3)) # GAP
        output = self.classifier(features).view(b)
        return output