import argparse
import torchvision

from losses import W1, W2
from models import get_generator
from utils import *

torch.backends.cudnn.benchmark = True



def OTS(netG, psi, opt_psi, data, loss_func, args, it, device):
    ot_loss = []
    w1_estimate = []
    memory_z = []
    memory_idx = []
    for ots_iter in range(args.max_ots_iters):
        opt_psi.zero_grad()

        # print(f"{ots_iter}: {human_format(torch.cuda.memory_allocated(0))}")

        z_batch = torch.randn(args.batch_size, args.nz, device=device)
        y_fake = netG(z_batch).detach()

        phi, idx = loss_func.score(y_fake, data, psi)

        loss = torch.mean(phi) + torch.mean(psi)
        loss_primal = loss_func.loss(y_fake, data[idx])

        loss_back = -torch.mean(psi[idx])  # equivalent to loss
        loss_back.backward()

        opt_psi.step()

        ot_loss.append(loss.item())
        w1_estimate.append(loss_primal.item())

        memory_z.append(z_batch)
        memory_idx.append(idx)
        if len(memory_z) > args.memory_size:
            memory_z = memory_z[1:]
            memory_idx = memory_idx[1:]

        if ots_iter % 1000 == 0:
            # Empirical stopping criterion
            np_indices = torch.cat(memory_idx).reshape(-1).cpu().numpy()
            n, min_f, max_f = args.early_end
            histo = np.histogram(np_indices, bins=n, range=(0, len(data) - 1), density=True)[0]
            histo /= histo.sum()
            if histo.min() >= min_f/n and histo.max() <= max_f/n:
                break
            print(f"Iteration {it}, OTS: {ots_iter}, "
                  f"loss_dual: {np.mean(ot_loss[-1000:]):.2f}, "
                  f"loss_primal: {np.mean(w1_estimate[-1000:]):.2f}, "
                  f"histogram ({histo.min()}: {histo.max()}), goal ({min_f/n}, {max_f/n}), "
                  f"memory-size: {len(memory_z)}, " +
                  (f"Cuda-memory {human_format(torch.cuda.memory_allocated(0))}" if device == torch.device("cuda") else ""))

    memory = list(zip(memory_z, memory_idx))
    return memory


def FIT(netG, opt_g, data, memory, loss_func):
    g_loss = []
    for z_batch, data_idx in memory:
        opt_g.zero_grad()

        y_fake = netG(z_batch)  # G_t(z)

        loss_g = loss_func.loss(data[data_idx], y_fake)

        loss_g.backward()
        opt_g.step()

        g_loss.append(loss_g.item())

    # netG.zero_grad()
    return g_loss


def train_images(args):
    os.makedirs(output_dir, exist_ok=True)
    data = get_data(args.data_path, args.im_size, gray=False).to(device)
    output_dim = args.c * args.im_size**2
    data = data.view(len(data), output_dim)

    loss_func = W1() if args.distance == "W1" else W2()

    netG = get_generator(args.nz, args.n_hidden, output_dim).to(device)
    opt_g = torch.optim.Adam(netG.parameters(), lr=1e-4)

    psi = torch.zeros(len(data), requires_grad=True, device=device)
    opt_psi = torch.optim.Adam([psi], lr=1e-1)

    debug_fixed_z = torch.randn(args.batch_size, args.nz, device=device)
    print(f"- , {torch.cuda.memory_allocated(0)}")

    for it in range(args.n_iters):
        memory = OTS(netG, psi, opt_psi, data, loss_func, args, it, device)
        g_loss = FIT(netG, opt_g, data, memory, loss_func)

        print(it, "FIT loss:", np.mean(g_loss))
        y_output = netG(debug_fixed_z).view(-1, args.c, args.im_size, args.im_size)
        torchvision.utils.save_image(y_output, f"{output_dir}/fake_%d.png" % it, normalize=True)


def train_patches(args):
    p = args.p
    s = args.s
    d = args.im_size
    c = args.c
    os.makedirs(output_dir, exist_ok=True)
    data = get_data(args.data_path, args.im_size, gray=False).to(device)
    data = to_patches(data, d, c, p, s)
    print(f"Got {len(data)} patches")

    loss_func = W1() if args.distance == "W1" else W2()

    netG = get_generator(args.nz, args.n_hidden, output_dim=c*d**2).to(device)
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
    args.data_path = "/mnt/storage_ssd/datasets/FFHQ/FFHQ_128/FFHQ_128"
    args.c = 3

    args.im_size = 64
    args.nz = 64

    args.batch_size = 64
    args.n_hidden = 512

    # Empirical stopping criterion
    args.memory_size = 4000
    args.early_end = (1000, 200, 320)

    args.n_iters = 100
    args.max_ots_iters = 20000

    args.distance = "W2"  # W1 or W2 or hybrid: W1 better looking, W2 faster

    output_dir = f"{os.path.basename(args.data_path)}_I-{args.im_size}"
    train_images(args)

    # args.p = 32
    # args.s = 16
    # output_dir += f"_P-{args.p}_S-{args.s}"
    # train_patches(args)





