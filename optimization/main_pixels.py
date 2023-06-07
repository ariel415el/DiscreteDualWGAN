import sys
import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torchvision

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from losses import *
from utils import *

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)


def OTS(outputs, psi, opt_psi, data, loss_func, args, it, patch_mode):
    w1_estimate = []
    idx_memory_outputs = []
    idx_memory_data = []
    if patch_mode:
        outputs = to_patches(outputs, args.im_size, args.c, p=args.p, s=args.s)
    for ots_iter in range(args.max_ots_iters):
        opt_psi.zero_grad()

        data_idx = torch.randperm(len(data))[:args.batch_size]

        phi, outputs_idx = loss_func.score(data[data_idx], outputs, psi)

        dual_estimate = torch.mean(phi) + torch.mean(psi)

        loss = -1 * dual_estimate  # maximize over psi
        loss.backward()

        opt_psi.step()

        w1_estimate.append(dual_estimate.item())

        idx_memory_outputs.append(outputs_idx)
        idx_memory_data.append(data_idx)
        if len(idx_memory_outputs) > args.memory_size:
            idx_memory_outputs = idx_memory_outputs[1:]
            idx_memory_data = idx_memory_data[1:]

        if ots_iter % 100 == 0:
            np_indices = torch.cat(idx_memory_outputs).reshape(-1).cpu().numpy()
            n, min_f, max_f = args.early_end
            histo = np.histogram(np_indices, bins=n, range=(0, len(data) - 1), density=True)[0]
            histo /= histo.sum()
            if histo.min() >= min_f/n and histo.max() <= max_f/n:
                break
            print(f"Iteration {it}, OTS: {ots_iter}, "
                  f"w1_estimate: {np.mean(w1_estimate[-1000:]):.2f}, "
                  f"histogram ({histo.min()}: {histo.max()}), goal ({min_f/n}, {max_f/n}), "
                  f"memory-size: {len(idx_memory_outputs)}, " +
                  (f"Cuda-memory {human_format(torch.cuda.memory_allocated(0))}" if device == torch.device("cuda") else ""))

    memory = list(zip(idx_memory_data, idx_memory_outputs))
    return memory


def FIT(outputs, opt, data, memory, loss_func, patch_mode):
    opt.zero_grad()
    losses = []
    total_loss = 0
    for data_idx, outputs_idx in memory:
        if patch_mode:
            outputs_ = to_patches(outputs, args.im_size, args.c, p=args.p, s=args.s)
        else:
            outputs_ = outputs
        opt.zero_grad()

        loss = loss_func.loss(data[data_idx], outputs_[outputs_idx])
        # total_loss += loss

        loss.backward()
        opt.step()

        losses.append(loss.item())
    # total_loss.backward()
    # opt.step()
    return losses


def train(args, patch_mode=False):
    os.makedirs(output_dir, exist_ok=True)

    loss_func = get_loss_function(args.loss_name)

    # Load data
    data = get_data(args.data_path, args.im_size, args.gray).to(device)
    args.c = data.shape[1]

    # Init optimized images
    if args.init_mode == "zeros":
        outputs = torch.zeros(args.num_images,  args.c, args.im_size, args.im_size).to(device)
    elif args.init_mode == "mean":
        outputs = torch.mean(data, dim=0, keepdim=True).repeat(args.num_images, 1, 1, 1)
    else:
        outputs = torch.load(args.init_mode).to(device).detach()

    outputs += torch.rand_like(outputs) * args.init_noise

    outputs.requires_grad_()
    opt_outputs = torch.optim.Adam([outputs], lr=args.lr_I)

    # Reshape data and outputs
    if patch_mode:
        data = to_patches(data, args.im_size, args.c, p=args.p, s=args.s)

        n_patches_in_image = compute_n_patches_in_image(args.im_size, args.c, args.p, args.s)
        args.batch_size *= n_patches_in_image
        n_duals = args.num_images * n_patches_in_image
    else:
        data = data.view(len(data), -1)
        outputs = outputs.view(len(outputs), -1)
        n_duals = len(outputs)

    psi = torch.ones(n_duals, requires_grad=True, device=device)
    opt_psi = torch.optim.Adam([psi], lr=args.lr_phi)

    print(f"Data size: {data.shape}, Outputs size: {outputs.shape}")
    plot = []
    for it in range(args.n_iters):
        memory = OTS(outputs.detach(), psi, opt_psi, data, loss_func, args, it, patch_mode)
        g_loss = FIT(outputs, opt_outputs, data, memory, loss_func, patch_mode)
        plot.append(np.mean(g_loss))
        print(it, "FIT loss:", np.mean(g_loss))
        if patch_mode:
            images = outputs
        else:
            images = outputs.reshape(-1, args.c, args.im_size, args.im_size)
        torch.save(images.detach(), f'{output_dir}/outputs.pt')
        torchvision.utils.save_image(images, f"{output_dir}/fake_%d.png" % it, normalize=True)

        plt.plot(range(len(plot)), plot)
        plt.savefig((f"{output_dir}/plot.png"))
        plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data_path', default="/mnt/storage_ssd/datasets/FFHQ/FFHQ_128/FFHQ_128", help="Path to train images")
    parser.add_argument('--gray', default=False, action='store_true', help="Convert images to grayscale")

    # Model
    parser.add_argument('--num_images', default=64, type=int)
    parser.add_argument('--im_size', default=64, type=int)
    parser.add_argument('--init_mode', default='zeros', type=str)
    parser.add_argument('--init_noise', default=1, type=float)
    parser.add_argument('--p', default=None, type=int)
    parser.add_argument('--s', default=None, type=int)

    # Training
    parser.add_argument('--loss_name', default='W1', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr_I', default=0.001, type=float)
    parser.add_argument('--lr_phi', default=0.001, type=float)
    parser.add_argument('--n_iters', default=100, type=int)
    parser.add_argument('--memory_size', default=4000, type=int)
    parser.add_argument('--max_ots_iters', default=20000, type=int)
    parser.add_argument('--early_end', default=(1000, 0.5, 1.5), type=tuple)

    # Other
    parser.add_argument('--tag', default='test')
    parser.add_argument('--device', default="cuda:0")

    args = parser.parse_args()

    output_dir = f"outputs/{os.path.basename(args.data_path)}_I-{args.im_size}_{args.tag}_D-{args.loss_name}"

    device = torch.device(args.device)

    if args.p is not None:
        output_dir += f"_P-{args.p}_S-{args.s}"
        if args.s is None:
            args.s = args.p
        train(args, patch_mode=True)
    else:
        train(args)





