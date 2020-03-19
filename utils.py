import torchvision.transforms as transforms
import torch
import numpy as np
import torch.nn as nn


def transform_data(data, opt):
    data = torch.FloatTensor(data.astype("float"))
    transform_list = [
        transforms.ToPILImage(),
        transforms.Resize(int(np.sqrt(opt.input_dim))),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]

    transform = transforms.Compose(transform_list)
    return transform(data)


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
