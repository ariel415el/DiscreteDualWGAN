import os

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
            img = self.transforms(img).reshape(-1)
        return img, idx


class MemoryDataset(Dataset):
    def __init__(self, paths, im_size, gray=False):
        super(MemoryDataset, self).__init__()
        self.paths = paths
        transforms = get_transforms(im_size, gray)
        self.images = []
        for path in tqdm(paths):
            img = Image.open(path).convert('RGB')
            if transforms is not None:
                self.images.append(transforms(img))

        self.images = torch.stack(self.images).reshape(len(self.images), -1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.images[idx], idx

def get_dataloader(root, im_size, gray, batch_size, n_workers,  limit_data=None):
    paths = [os.path.join(root, x) for x in os.listdir(root)]
    if limit_data is not None:
        paths = paths[:limit_data]
    dataset = MemoryDataset(paths, im_size, gray=False)
    return DataLoader(dataset, batch_size=batch_size,
               shuffle=True,
               num_workers=n_workers,
               pin_memory=True)


def compute_n_patches_in_image(d, c, p, s):
    dummy_img = torch.zeros((1, c, d, d))
    dummy_patches = F.unfold(dummy_img, kernel_size=p, stride=s)
    patches_per_image = dummy_patches.shape[-1]  # shape (1, c*p*p, N_patches)
    return patches_per_image


