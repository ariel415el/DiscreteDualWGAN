import os

import numpy as np
import torch
from torch.nn import functional as F

from utils.data import compute_n_patches_in_image


def to_patches(x, d, c, p=8, s=4):
    patches = F.unfold(x.view(-1, c, d, d), kernel_size=p, stride=s)  # shape (b, c*p*p, N_patches)
    patches = patches.permute(0, 2, 1)               # shape (b, N_patches, c*p*p)
    patches = patches.reshape(-1, patches.shape[-1]) # shape (b * N_patches, c*p*p))

    return patches


def patches_to_image(patches, d, c, p=8, s=4):
    patches_per_image = compute_n_patches_in_image(d, c, p, s)

    patches = patches.reshape(-1, patches_per_image, c * p ** 2)
    patches = patches.permute(0, 2, 1)
    img = F.fold(patches, (d, d), kernel_size=p, stride=s)

    # normal fold matrix
    input_ones = torch.ones((1, c, d, d), dtype=patches.dtype, device=patches.device)
    divisor = F.unfold(input_ones, kernel_size=p, dilation=(1, 1), stride=s, padding=(0, 0))
    divisor = F.fold(divisor, output_size=(d, d), kernel_size=p, stride=s)

    divisor[divisor == 0] = 1.0
    return (img / divisor)


def get_centroids(data_torch, n_centroids, fname, use_faiss=True, force_recompute=False):
    print(data_torch.shape)
    data = data_torch.cpu().numpy()
    if os.path.exists(fname) and not force_recompute:
        centroids = np.load(fname)
    else:
        print(f"Learning Kmeans on data {data.shape}")
        if use_faiss:
            import faiss
            kmeans = faiss.Kmeans(data.shape[1], n_centroids, niter=100, verbose=False, gpu=True)
            kmeans.train(data)
            centroids = kmeans.centroids
        else:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_centroids, random_state=0, verbose=0).fit(data)
            centroids = kmeans.cluster_centers_

        np.save(fname, centroids)
    return torch.from_numpy(centroids).to(data_torch.device)
