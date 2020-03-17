from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import datasets
import torch
# from in_out.polygons import GenerateDataset
import os
import numpy as np


def get_dataloader(opt, return_test=False):
    if opt.dataset == "mnist":
        os.makedirs("in_out/storage/mnist", exist_ok=True)
        img_size = int(np.sqrt(opt.input_dim))
        dataloader_train = torch.utils.data.DataLoader(
            datasets.MNIST(
                "in_out/storage/mnist",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=opt.batch_size,
            shuffle=True,
        )

        dataloader_test = torch.utils.data.DataLoader(
            datasets.MNIST(
                "storage/mnist",
                train=False,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=opt.batch_size,
            shuffle=True,
        )

    elif opt.dataset == "polygons":
        dataopt = {"n_instances": 4094, "n_test": 128, "n_vertices": 3, "raster_dim": 64, "min_angle": 20}
        path = f"in_out/storage/{opt.dataset}"
        n = dataopt["n_instances"]
        if os.path.exists(path + f"/train_{n}_{opt.experiment}.npy"):
            print("loading data from memory")
            polygons_train = np.load(path + f"/train_{n}_{opt.experiment}.npy")
            polygons_dataset_train = CustomDataset(polygons_train, opt)
            dataloader_train = DataLoader(polygons_dataset_train, batch_size=opt.batch_size, shuffle=True)

            polygons_test = np.load(path + f"/test_{n}_{opt.experiment}.npy")
            polygons_dataset_test = CustomDataset(polygons_test, opt)
            dataloader_test = DataLoader(polygons_dataset_test, batch_size=opt.batch_size, shuffle=True)

        else:
            print("generating data")
            if opt.experiment == "triangles" or opt.experiment == "squares":
                os.makedirs("in_out/storage/polygons", exist_ok=True)

                # note n_instances should be divisable by the batch size
                polygons, labels = GenerateDataset(
                    n_instances=dataopt["n_instances"] + dataopt["n_test"],
                    n_vertices=dataopt["n_vertices"],
                    min_segment_angle=dataopt["min_angle"],
                    scale=0.75,
                    raster_dim=dataopt["raster_dim"],
                    subpixel_res=8,
                    shift_to_mean=True,
                    seed=0,
                )

                polygons_train = polygons[0:dataopt["n_instances"]]
                polygons_test = polygons[-dataopt["n_test"]:]

                np.save(path + f"/train_{n}_{opt.experiment}.npy", polygons_train)
                np.save(path + f"/test_{n}_{opt.experiment}.npy", polygons_test)

            elif opt.experiment == "mixed":
                n_instance_each = int((dataopt["n_instances"] + dataopt["n_test"])/ 2)

                triangles, _ = GenerateDataset(
                    n_instances=n_instance_each,
                    n_vertices=3,
                    min_segment_angle=dataopt["min_angle"],
                    scale=0.75,
                    raster_dim=dataopt["raster_dim"],
                    subpixel_res=8,
                    shift_to_mean=True,
                    seed=0,
                )

                squares, _ = GenerateDataset(
                    n_instances=n_instance_each,
                    n_vertices=4,
                    min_segment_angle=dataopt["min_angle"],
                    scale=0.75,
                    raster_dim=dataopt["raster_dim"],
                    subpixel_res=8,
                    shift_to_mean=True,
                    seed=0,
                )

                polygons = np.concatenate((triangles, squares))
                np.random.shuffle(polygons)

                polygons_train = polygons[0:dataopt["n_instances"]]
                polygons_test = polygons[-dataopt["n_test"]:]

                np.save(path + f"/train_{n}_{opt.experiment}.npy", polygons_train)
                np.save(path + f"/test_{n}_{opt.experiment}.npy", polygons_test)

            polygons_dataset_train = CustomDataset(polygons_train, opt)
            polygons_dataset_test = CustomDataset(polygons_test, opt)
            dataloader_train = DataLoader(polygons_dataset_train, batch_size=opt.batch_size, shuffle=True)
            dataloader_test = DataLoader(polygons_dataset_test, batch_size=opt.batch_size, shuffle=True)

    elif opt.dataset == "dsprites":
        path = "./in_out/storage/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
        data = SpritesDataset(path)
        dataloader_train = DataLoader(data, batch_size=opt.batch_size, shuffle=True)

    else:
        raise NotImplementedError("Choose a MNIST or as polygons as your dataset")

    if not return_test:
        dataloader_test = None

    return dataloader_train, dataloader_test


class CustomDataset(Dataset):
    """ Generated dataset of triangles """

    def __init__(self, data, opt):
        # self.data = torch.FloatTensor(data.astype("float"))
        self.data = data

        transform_list = [transforms.ToPILImage(),
                          transforms.Resize(int(np.sqrt(opt.input_dim))),
                          transforms.ToTensor(),
                          transforms.Normalize([0.5], [0.5])]

        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        data = self.transform(data)

        return data


class SpritesDataset(Dataset):
    def __init__(self, dataset_path):
        dataset_zip = np.load(dataset_path, encoding='latin1')
        self.imgs = dataset_zip['imgs']

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        return self.imgs[idx].astype(np.float32)



