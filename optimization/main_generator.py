import argparse
import os
import sys

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from losses import L1, L2
from models import FCGenerator, PixelGenerator
from utils import *
from utils.common import human_format
from utils.data import get_data
from utils.image import to_patches

torch.backends.cudnn.benchmark = True

z_batch = None

def OTS(netG, psi, opt_psi, data, loss_func, args, it, device):
    primals = []
    duals = []
    memory = []
    for ots_iter in range(args.max_ots_iters):
        opt_psi.zero_grad()

        # print(f"{ots_iter}: {human_format(torch.cuda.memory_allocated(0))}")

        global z_batch
        if z_batch is None:
            z_batch = torch.randn(args.batch_size, args.nz, device=device)

        y_fake = netG(z_batch).detach()

        phi, idx = loss_func.score(y_fake, data, psi)

        dual_loss = torch.mean(phi) + torch.mean(psi)
        primal_loss = loss_func.loss(y_fake, data[idx])
        loss_back = -1 * dual_loss
        # loss_back = -torch.mean(psi[idx])  # equivalent to loss
        loss_back.backward()

        opt_psi.step()

        duals.append(dual_loss.item())
        primals.append(primal_loss.item())

        memory.append((z_batch, idx))
        if len(memory) > args.memory_size:
            memory = memory[1:]

        if ots_iter % 100 == 0:
            # Empirical stopping criterion
            # np_indices = torch.cat(memory_idx).reshape(-1).cpu().numpy()
            # n, min_f, max_f = args.early_end
            # histo = np.histogram(np_indices, bins=n, range=(0, len(data) - 1), density=True)[0]
            # histo /= histo.sum()
            # if histo.min() >= min_f/n and histo.max() <= max_f/n:
            #     break
            print(f"Iteration {it}, OTS: {ots_iter}, " +
                  f"loss_dual: {duals[-1]:.2f}, " +
                  f"loss_primal: {primals[-1]:.2f}, " +
                  # f"histogram ({histo.min()}: {histo.max()}), goal ({min_f/n}, {max_f/n}), "
                  f"memory-size: {len(memory)}, " +
                  (f"Cuda-memory {human_format(torch.cuda.memory_allocated(0))}" if device == torch.device("cuda") else ""))

    return memory, primals, duals


def FIT(netG, opt_g, data, memory, loss_func):
    primals = []
    for z_batch, data_idx in memory:
        opt_g.zero_grad()

        y_fake = netG(z_batch)  # G_t(z)

        primal_loss = loss_func.loss(data[data_idx], y_fake)

        primal_loss.backward()
        opt_g.step()

        primals.append(primal_loss.item())

    return primals


def train_images(args):
    os.makedirs(output_dir, exist_ok=True)
    data = get_data(args.data_path, args.im_size, gray=False, limit_data=args.limit_data, center_crop=args.center_crop).to(device)
    output_dim = args.c * args.im_size**2
    data = data.view(len(data), output_dim)

    loss_func = L1() if args.distance == "L1" else L2()

    if args.arch == "FC":
        netG = FCGenerator(args.nz, output_dim, args.n_hidden).to(device)
    else:
        netG = PixelGenerator(args.nz, args.im_size).to(device)
    opt_g = torch.optim.Adam(netG.parameters(), lr=0.001)

    psi = torch.randn(len(data), requires_grad=True, device=device)
    opt_psi = torch.optim.Adam([psi], lr=0.01)

    debug_fixed_z = torch.randn(args.batch_size, args.nz, device=device)
    print(f"- , {torch.cuda.memory_allocated(0)}")

    all_primals = []
    all_duals = []
    for it in range(args.n_iters):
        memory, primals, duals = OTS(netG, psi, opt_psi, data, loss_func, args, it, device)
        g_loss = FIT(netG, opt_g, data, memory, loss_func)
        print(it, "FIT loss:", np.mean(g_loss))

        all_duals += duals
        all_primals += primals
        plt.plot(range(len(all_primals)), all_primals, label="Primal", color='r')
        plt.title(f"Last: {all_primals[-1]:.4f}" )
        plt.plot(range(len(all_duals)), all_duals, label="Dual", color='b')
        plt.title(f"Last: {all_duals[-1]:.4f}" )
        plt.legend()
        plt.savefig(f"{output_dir}/plot.png")
        plt.clf()

        y_output = netG(debug_fixed_z).view(-1, args.c, args.im_size, args.im_size)
        torchvision.utils.save_image(y_output, f"{output_dir}/fixed_z_fake_%d.png" % it, normalize=True)
        y_output = netG(torch.randn(args.batch_size, args.nz, device=device)).view(-1, args.c, args.im_size, args.im_size)
        torchvision.utils.save_image(y_output, f"{output_dir}/fake_%d.png" % it, normalize=True)


def train_patches(args):
    p = args.p
    s = args.s
    d = args.im_size
    c = args.c
    os.makedirs(output_dir, exist_ok=True)
    data = get_data(args.data_path, args.im_size, gray=False, limit_data=10000).to(device)
    data = to_patches(data, d, c, p, s)
    print(f"Got {len(data)} patches")

    loss_func = L1() if args.distance == "L2" else L2()

    # netG = get_generator(args.nz, args.n_hidden, output_dim=c*d**2).to(device)
    netG = PixelGenerator(args.nz, args.im_size).to(device)

    opt_g = torch.optim.Adam(netG.parameters(), lr=1e-4)

    def patch_wrapper(x):
        return to_patches(netG(x), d, c, p, s)

    psi = torch.zeros(len(data), requires_grad=True, device=device)
    opt_psi = torch.optim.Adam([psi], lr=1e-1)

    debug_fixed_z = torch.randn(args.batch_size, args.nz, device=device)
    for it in range(args.n_iters):
        memory = OTS(patch_wrapper, psi, opt_psi, data, loss_func, args, it, device)
        g_loss = FIT(patch_wrapper, opt_g, data, memory, loss_func)

        print(it, "FIT loss:", np.mean(g_loss))
        y_output = netG(debug_fixed_z).view(-1, args.c, args.im_size, args.im_size)
        torchvision.utils.save_image(y_output, f"{output_dir}/fake_%d.png" % it, normalize=True)


if __name__ == '__main__':
    device = torch.device("cuda")
    args = argparse.Namespace()
    args.data_path = "/mnt/storage_ssd/datasets/FFHQ/FFHQ/FFHQ"
    args.c = 3
    args.limit_data = 10000
    args.center_crop = 90

    args.im_size = 64
    args.nz = 64

    args.arch = "Pixels"
    args.batch_size = 64
    args.n_hidden = 128

    # Empirical stopping criterion
    args.memory_size = 2000
    args.early_end = (1000, 200, 320)

    args.n_iters = 100
    args.max_ots_iters = 2000

    args.distance = "L2"  # W1 or W2 or hybrid: W1 better looking, W2 faster

    output_dir = f"{os.path.basename(args.data_path)}_I-{args.im_size}"
    train_images(args)




