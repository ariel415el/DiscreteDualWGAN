import os

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import transforms


def get_data(root, im_size, gray, limit_data=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(im_size),
        transforms.Normalize((0.5,), (0.5,))
    ])

    images = []
    paths = [os.path.join(root, x) for x in os.listdir(root)]
    if limit_data is not None:
        paths = paths[:limit_data]
    for path in tqdm(paths, desc="Loading images into memory"):
        img = Image.open(path).convert('RGB')
        img = transform(img)
        images.append(img)

    data = torch.stack(images)
    if gray:
        data = torch.mean(data, dim=1, keepdim=True)

    return data


def get_transforms(im_size, gray):
    ts = []
    if gray:
        ts.append(transforms.Grayscale())
    ts += [
        transforms.ToTensor(),
        transforms.Resize(im_size),
        transforms.Normalize((0.5,), (0.5,))
    ]
    return transforms.Compose(ts)


class DiskDataset(Dataset):
    def __init__(self, paths, im_size, gray=False):
        super(DiskDataset, self).__init__()
        self.paths = paths
        self.transforms = get_transforms(im_size, gray)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img, idx


def get_dataloader(root, im_size, gray, batch_size, n_workers,  limit_data=None):
    paths = [os.path.join(root, x) for x in os.listdir(root)]
    if limit_data is not None:
        paths = paths[:limit_data]
    dataset = DiskDataset(paths, im_size, gray=False)
    return DataLoader(dataset, batch_size=batch_size,
               shuffle=True,
               num_workers=n_workers,
               pin_memory=True)


def to_patches(x, d, c, p=8, s=4):
    xp = x.reshape(-1, c, d, d)  # shape  (b,c,d,d)
    patches = F.unfold(xp, kernel_size=p, stride=s)  # shape (b, c*p*p, N_patches)
    patches = patches.permute(0, 2, 1)               # shape (b, N_patches, c*p*p)
    patches = patches.reshape(-1, patches.shape[-1]) # shape (b * N_patches, c*p*p))

    return patches


def compute_n_patches_in_image(d, c, p, s):
    dummy_img = torch.zeros((1, c, d, d))
    dummy_patches = F.unfold(dummy_img, kernel_size=p, stride=s)
    patches_per_image = dummy_patches.shape[-1]  # shape (1, c*p*p, N_patches)
    return patches_per_image


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


def human_format(num):
    """
    :param num: A number to print in a nice readable way.
    :return: A string representing this number in a readable way (e.g. 1000 --> 1K).
    """
    magnitude = 0

    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0

    return '%.3f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])  # add more suffices if you n


def parse_classnames_and_kwargs(string, kwargs=None):
    """Import a class and and create an instance with kwargs from strings in format
    '<class_name>_<kwarg_1_name>=<kwarg_1_value>_<kwarg_2_name>=<kwarg_2_value>"""
    name_and_args = string.split('-')
    class_name = name_and_args[0]
    if kwargs is None:
        kwargs = dict()
    for arg in name_and_args[1:]:
        name, value = arg.split("=")
        kwargs[name] = eval(value)
    return class_name, kwargs


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