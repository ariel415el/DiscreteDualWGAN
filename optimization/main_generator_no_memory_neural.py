import argparse
import os
import sys
from time import time

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from losses import L1, L2
from models import FCDiscriminator, PixelGenerator, FCGenerator, DCDiscriminator, DCGenerator
from utils import *
from utils.common import human_format
from utils.data import get_data
from utils.image import to_patches

torch.backends.cudnn.benchmark = True


def OTS(netG, psi_func, opt_psi, data, loss_func, args, it, device):
    primals = []
    duals = []
    for ots_iter in range(args.n_OTS_steps):
        opt_psi.zero_grad()

        z_batch = torch.randn(args.batch_size, args.nz, device=device)
        y_fake = netG(z_batch).detach()

        psi = get_psi_on_data(psi_func, data)
        phi, idx = loss_func.score(y_fake, data, psi)

        dual_loss = torch.mean(phi) + torch.mean(psi)
        loss = -1 * dual_loss
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


def get_psi_on_data(psi_func, data, b=512):
    start = time()
    n = len(data)
    psi = torch.zeros(n).to(data.device)
    n_batches = n // b
    for i in range(n_batches):
        s = slice(i * b,(i + 1) * b)
        psi[s] = psi_func(data[s])
    if n % b != 0:
        s = slice(n_batches * b, n)
        psi[s] = psi_func(data[s])

    # print(f"{n} potentials computed in  {time()-start} seconds")
    return psi

def train_images(args):
    os.makedirs(output_dir, exist_ok=True)
    data = get_data(args.data_path, args.im_size, gray=False, limit_data=args.limit_data).to(device)
    args.c = data.shape[1]
    output_dim = args.c * args.im_size**2
    data = data.view(len(data), output_dim)

    loss_func = L1() if args.distance == "L1" else L2()

    if args.gen_arch == "FC":
        netG = FCGenerator(args.nz, output_dim, args.n_hidden).to(device)
    elif args.gen_arch == "DC":
        netG = DCGenerator(args.nz, args.n_hidden).to(device)
    else:
        netG = PixelGenerator(args.nz, args.im_size).to(device)
        args.lr_G *= 100

    if args.disc_arch == "FC":
        psi_func = FCDiscriminator(64, args.n_hidden).to(device)
    elif args.disc_arch == "DC":
        psi_func = DCDiscriminator(64, args.n_hidden).to(device)
    else:
        psi_func = DCDiscriminator(64, args.n_hidden).to(device)

    opt_g = torch.optim.Adam(netG.parameters(), lr=args.lr_G)
    opt_psi = torch.optim.Adam(psi_func.parameters(), lr=args.lr_psi)

    debug_fixed_z = torch.randn(args.batch_size, args.nz, device=device)
    print(f"- , {torch.cuda.memory_allocated(0)}")

    all_primals = []
    all_primals2 = []
    all_duals = []
    for it in range(args.n_iters):
        primals, duals = OTS(netG, psi_func, opt_psi, data, loss_func, args, it, device)
        all_primals += primals
        all_duals += duals

        psi = get_psi_on_data(psi_func, data)
        primals, duals = FIT(netG, opt_g, data, psi, loss_func)
        all_primals += primals
        all_duals += duals
        print(it, f"FIT: primal avg: {np.mean(primals)} Dual avg {np.mean(duals)}", "Dual avg")

        plt.plot(range(len(all_primals)), all_primals, label="Primal", color='r')
        plt.title(f"Last: {all_primals[-1]:.4f}" )
        plt.plot(range(len(all_duals)), all_duals, label="Dual", color='b')
        plt.title(f"Last: {all_duals[-1]:.4f}" )
        if all_primals2:
            plt.plot(np.linspace(0,len(all_primals), len(all_primals2)), all_primals2, label="OT", color='g')
            plt.title(f"Last: {all_primals2[-1]:.4f}" )
        plt.legend()
        plt.savefig(f"{output_dir}/plot.png")
        plt.clf()

        if it % args.save_freq == 0:
            z_batch = torch.randn(512, args.nz, device=device)
            y_fake = netG(z_batch)  # G_t(z)
            primal = primal_ot(y_fake, data)
            all_primals2.append(primal)

            y_output = netG(debug_fixed_z).view(-1, args.c, args.im_size, args.im_size)
            torchvision.utils.save_image(y_output, f"{output_dir}/fixed_z_fake_%d.png" % it, normalize=True)
            y_output = netG(torch.randn(args.batch_size, args.nz, device=device)).view(-1, args.c, args.im_size, args.im_size)
            torchvision.utils.save_image(y_output, f"{output_dir}/fake_%d.png" % it, normalize=True)

def primal_ot(X, Y):
    from losses import batch_dist_matrix, efficient_L2_dist_matrix
    import ot
    uniform_x = np.ones(len(X)) / len(X)
    uniform_y = np.ones(len(Y)) / len(Y)
    C = batch_dist_matrix(X, Y, b=512, dist_function=efficient_L2_dist_matrix)
    OT = ot.emd2(uniform_x, uniform_y, C.cpu().detach().numpy())
    return OT

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data_path', default="/mnt/storage_ssd/datasets/FFHQ/FFHQ_128/FFHQ_128", help="Path to train images")
    parser.add_argument('--gray', default=False, action='store_true', help="Convert images to grayscale")
    parser.add_argument('--limit_data', default=10000, type=int)

    # Model
    parser.add_argument('--gen_arch', default='Pixels', type=str, help="FC or pixels")
    parser.add_argument('--disc_arch', default='DC', type=str, help="FC or pixels")
    parser.add_argument('--im_size', default=64, type=int)
    parser.add_argument('--p', default=None, type=int)
    parser.add_argument('--s', default=None, type=int)
    parser.add_argument('--nz', default=64, type=int)
    parser.add_argument('--n_hidden', default=32, type=int)

    # Training
    parser.add_argument('--distance', default='L2', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr_G', default=0.001, type=float)
    parser.add_argument('--lr_psi', default=0.001, type=float)
    parser.add_argument('--n_iters', default=2000, type=int)
    parser.add_argument('--n_OTS_steps', default=100, type=int)
    parser.add_argument('--n_FIT_steps', default=1, type=int)

    # Other
    parser.add_argument('--tag', default='test')
    parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--device', default="cuda:0")

    args = parser.parse_args()
    device = torch.device(args.device)


    output_dir = f"outputs/{os.path.basename(args.data_path)}_I-{args.im_size}_{args.tag}_L-{args.distance}_G-{args.gen_arch}_D-{args.disc_arch}"

    train_images(args)





