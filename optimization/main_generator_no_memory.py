import argparse
import os
import sys

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from losses import L1, L2
from models import get_generator, PixelGenerator
from utils import *
from utils.common import human_format
from utils.data import get_data
from utils.image import to_patches

torch.backends.cudnn.benchmark = True


def OTS(netG, psi, opt_psi, data, loss_func, args, it, device):
    primals = []
    duals = []
    memory = []
    for ots_iter in range(args.n_OTS_steps):
        opt_psi.zero_grad()

        # print(f"{ots_iter}: {human_format(torch.cuda.memory_allocated(0))}")

        z_batch = torch.randn(args.batch_size, args.nz, device=device)
        y_fake = netG(z_batch).detach()

        phi, idx = loss_func.score(y_fake, data, psi)

        dual_loss = torch.mean(phi) + torch.mean(psi)
        loss = -1 * dual_loss
        # loss = -torch.mean(psi[idx])  # equivalent to loss
        loss.backward()

        opt_psi.step()

        duals.append(dual_loss.item())
        primal = loss_func.loss(y_fake, data[idx])
        primals.append(primal.item())

        # memory.append((z_batch, idx))
        if ots_iter % 100 == 0:
            # if histo.min() >= min_f/n and histo.max() <= max_f/n:
            #     break
            print(f"Iteration {it}, OTS: {ots_iter}, "
                  f"last dual-loss: {np.mean(duals[-1:]):.2f}, "
                  f"last primal-loss: {np.mean(primals[-1:]):.2f}, "+
                  (f"Cuda-memory {human_format(torch.cuda.memory_allocated(0))}" if device == torch.device("cuda") else ""))

    return primals, duals

def FIT(netG, opt_g, data, psi, loss_func):
    primals = []
    duals = []
    for _ in range(args.n_FIT_steps):
        opt_g.zero_grad()

        z_batch = torch.randn(args.batch_size, args.nz, device=device)
        y_fake = netG(z_batch)  # G_t(z)

        phi, data_idx = loss_func.score(y_fake, data, psi)

        primal = loss_func.loss(data[data_idx], y_fake)
        dual = torch.mean(phi) + torch.mean(psi)

        primal.backward()
        opt_g.step()

        primals.append(primal.item())
        duals.append(dual.item())

    # netG.zero_grad()
    return primals, duals

def FIT_memory(netG, opt_g, data, psi, loss_func, memory):
    primals = []
    duals = []
    for z_batch, data_idx in memory:
        opt_g.zero_grad()

        y_fake = netG(z_batch)  # G_t(z)

        primal = loss_func.loss(data[data_idx], y_fake)
        dual = torch.mean(phi) + torch.mean(psi)

        primal.backward()
        opt_g.step()

        primals.append(primal.item())
        duals.append(dual.item())

    # netG.zero_grad()
    return primals, duals


def train_images(args):
    os.makedirs(output_dir, exist_ok=True)
    data = get_data(args.data_path, args.im_size, gray=False, limit_data=args.limit_data).to(device)
    args.c = data.shape[1]
    output_dim = args.c * args.im_size**2
    data = data.view(len(data), output_dim)

    loss_func = L1() if args.distance == "L1" else L2()

    if args.arch == "FC":
        netG = get_generator(args.nz, 128, output_dim).to(device)
        args.lr_G *= 0.1
    else:
        netG = PixelGenerator(args.nz, args.im_size).to(device)
    opt_g = torch.optim.Adam(netG.parameters(), lr=args.lr_G)

    psi = torch.zeros(len(data), requires_grad=True, device=device)
    opt_psi = torch.optim.Adam([psi], lr=args.lr_psi)

    debug_fixed_z = torch.randn(args.batch_size, args.nz, device=device)
    print(f"- , {torch.cuda.memory_allocated(0)}")

    all_primals = []
    all_duals = []
    for it in range(args.n_iters):
        primals, duals = OTS(netG, psi, opt_psi, data, loss_func, args, it, device)
        all_primals += primals
        all_duals += duals

        primals, duals = FIT(netG, opt_g, data, psi, loss_func)
        all_primals += primals
        all_duals += duals
        print(it, f"FIT: primal avg: {np.mean(primals)} Dual avg {np.mean(duals)}", "Dual avg")

        plt.plot(range(len(all_primals)), all_primals, label="Primal", color='r')
        plt.title(f"Last: {all_primals[-1]:.4f}" )
        plt.plot(range(len(all_duals)), all_duals, label="Dual", color='b')
        plt.title(f"Last: {all_duals[-1]:.4f}" )
        plt.legend()
        plt.savefig(f"{output_dir}/plot.png")
        plt.clf()

        if it % args.save_freq == 0:
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
    data = get_data(args.data_path, args.im_size, gray=args.gray, limit_data=args.limit_data).to(device)
    args.c = data.shape[1]
    data = to_patches(data, d, c, p, s)
    print(f"Got {len(data)} patches")

    loss_func = L1() if args.distance == "L2" else L2()

    # netG = get_generator(args.nz, args.n_hidden, output_dim=c*d**2).to(device)
    netG = PixelGenerator(None, args.im_size).to(device)

    opt_g = torch.optim.Adam(netG.parameters(), lr=args.lr_G)

    def patch_wrapper(x):
        return to_patches(netG(x), d, c, p, s)

    psi = torch.zeros(len(data), requires_grad=True, device=device)
    opt_psi = torch.optim.Adam([psi], lr=args.lr_psi)

    debug_fixed_z = torch.randn(args.batch_size, args.nz, device=device)
    for it in range(args.n_iters):
        memory = OTS(patch_wrapper, psi, opt_psi, data, loss_func, args, it, device)
        g_loss = FIT(patch_wrapper, opt_g, data, memory, loss_func)

        print(it, "FIT loss:", np.mean(g_loss))
        plt.plot(range(len(g_loss)), g_loss)
        plt.title(f"FIT Loss: {g_loss[-1]:.5f}" )
        plt.savefig(f"{output_dir}/plot.png")
        plt.clf()

        y_output = netG(debug_fixed_z).view(-1, args.c, args.im_size, args.im_size)
        torchvision.utils.save_image(y_output, f"{output_dir}/fixed_z_fake_%d.png" % it, normalize=True)
        y_output = netG(torch.randn(args.batch_size, args.nz, device=device)).view(-1, args.c, args.im_size, args.im_size)
        torchvision.utils.save_image(y_output, f"{output_dir}/fake_%d.png" % it, normalize=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data_path', default="/mnt/storage_ssd/datasets/FFHQ/FFHQ_128/FFHQ_128", help="Path to train images")
    parser.add_argument('--gray', default=False, action='store_true', help="Convert images to grayscale")
    parser.add_argument('--limit_data', default=10000, type=int)

    # Model
    parser.add_argument('--arch', default='FC', type=str, help="FC or pixels")
    parser.add_argument('--im_size', default=64, type=int)
    parser.add_argument('--p', default=None, type=int)
    parser.add_argument('--s', default=None, type=int)
    parser.add_argument('--nz', default=64, type=int)

    # Training
    parser.add_argument('--distance', default='L2', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr_G', default=0.01, type=float)
    parser.add_argument('--lr_psi', default=0.1, type=float)
    parser.add_argument('--n_iters', default=2000, type=int)
    parser.add_argument('--n_OTS_steps', default=100, type=int)
    parser.add_argument('--n_FIT_steps', default=1, type=int)

    # Other
    parser.add_argument('--tag', default='test')
    parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--device', default="cuda:0")

    args = parser.parse_args()
    device = torch.device(args.device)


    output_dir = f"outputs/{os.path.basename(args.data_path)}_I-{args.im_size}_{args.tag}_D-{args.distance}"

    if args.p is not None:
        output_dir += f"_P-{args.p}_S-{args.s}"
        if args.s is None:
            args.s = args.p
        train_patches(args)
    else:
        train_images(args)





