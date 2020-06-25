from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import datasets
import torch
import os
import numpy as np


def get_dataloader(opt):
    if opt.dataset == "mnist":
        os.makedirs("../data/mnist", exist_ok=True)
        img_size = int(np.sqrt(opt.input_dim))
        dataloader_train = torch.utils.data.DataLoader(
            datasets.MNIST(
                "../data/mnist",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Resize(img_size),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ]
                ),
            ),
            batch_size=opt.batch_size,
            shuffle=True,
        )

        dataloader_test = torch.utils.data.DataLoader(
            datasets.MNIST(
                "../data//mnist",
                train=False,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Resize(img_size),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ]
                ),
            ),
            batch_size=opt.batch_size,
            shuffle=True,
        )

    elif opt.dataset == "dsprites":
        path = "../data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
        data = SpritesDataset(path)
        dataloader_train = DataLoader(data, batch_size=opt.batch_size, shuffle=True)

    else:
        raise NotImplementedError("Not implemented that dataset. Choose another.")

    return dataloader_train


class SpritesDataset(Dataset):
    def __init__(self, dataset_path):
        dataset_zip = np.load(dataset_path, encoding="latin1")
        self.imgs = dataset_zip["imgs"]

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        return self.imgs[idx].astype(np.float32)
