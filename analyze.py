from utils import transform_data
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader
from load_data import CustomDataset
import torch
from torchvision.utils import save_image


parser = argparse.ArgumentParser(description="Analyze a VAE model")
parser.add_argument(
    "--n_instances",
    type=int,
    default=1024,
    help="Size of dataset that the model is trained on",
)
parser.add_argument(
    "--dataset", type=str, default="polygons", help="which dataset to run"
)
parser.add_argument(
    "--input_dim", type=int, default=28 * 28, help="dimension of input space"
)
parser.add_argument(
    "--batch_size", type=int, default=64, help="number of datapoints in each batch"
)


def plot_loss(train):
    fig = plt.figure(figsize=(4 * 1.616, 4))
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    sns.set_palette("muted")
    for i, t in enumerate(train):
        elbo = [-x for x in t]
        plt.plot(elbo, label=2 ** (2 * i + 1))
    # plt.plot(test, label="Test", color="coral")
    plt.xlabel(r"Epoch", fontsize=15)
    plt.ylabel("ELBO", fontsize=15)
    plt.legend(fontsize=12)
    plt.show()


def analyze_prior():
    vae = model_data["model"]

    if os.path.exists(
            f"./in_out/storage/{opt.dataset}/test_{opt.n_instances}_{opt.experiment}.npy"
    ):
        test_data = np.load(
            f"./in_out/storage/{opt.dataset}/test_{opt.n_instances}_{opt.experiment}.npy"
        )
        test_labels = np.load(
            f"./in_out/storage/{opt.dataset}/test_{opt.n_instances}_{opt.experiment}_labels.npy"
        )
        test_dataset = CustomDataset(test_data, test_labels, opt)
        test_dataloader = DataLoader(
            test_dataset, batch_size=len(test_data), shuffle=True
        )
    else:
        raise OSError("Dataset is not stored.")

    with torch.no_grad():
        for _, (data, _) in enumerate(test_dataloader):
            mu, _ = vae.encode(data)

    mu = mu.numpy()
    x = mu[:, 0]
    y = mu[:, 1]
    sns.jointplot(x=x, y=y, kind="kde")
    plt.title("Encoded mean")
    plt.show()

    return


def walk_latent_space(row=8, interval=3):
    """
    :param row: number of rows in the image grid
    :param interval: number of standard deviation to sample from. Default is the
    :return: save an image grid
    """

    # assume prior is standard gaussian.
    z1 = np.linspace(start=-interval, stop=interval, num=row)
    z2 = np.linspace(start=-interval, stop=interval, num=row)
    z1_grid, z2_grid = np.meshgrid(z1, z2)
    z_grid = np.array((z1_grid.flatten(), z2_grid.flatten())).T
    vae = model_data["model"]
    vae.double()
    with torch.no_grad():
        samples = vae.decode(torch.from_numpy(z_grid))

    img_size = int(np.sqrt(opt.input_dim))
    save_image(
        samples.view(row * row, 1, img_size, img_size),
        f"./in_out/storage/results/{opt.experiment}_{opt.n_instances}/interpolations_{interval}.png",
        nrow=row,
    )


def explore_latent_variables(row=8, interval=3):
    """
    Sample two datapoints and interpolate in each individual latent variable
    :return:
    """
    # sample a points uniformly from test data
    torch.manual_seed(0)
    vae = model_data["model"]
    data = np.load(
        f"./in_out/storage/{opt.dataset}/test_{opt.n_instances}_{opt.experiment}.npy"
    )
    indices_choice = np.random.choice(len(data), size=1)
    points = data[indices_choice]
    points = transform_data(points, opt)

    # encode the chosen point
    mu, log_std = vae.encode(points)
    std = torch.exp(0.5 * log_std)
    epsilon = torch.randn_like(std)
    z = mu + epsilon * std
    z = z.detach().numpy()[0]
    latent_dim = z.shape[-1]

    # change each latent variable on uniform interval whole holding the others variable constant
    uniform_spacing = np.linspace(start=-interval, stop=interval, num=row)
    z_interpolates = np.zeros((latent_dim, row, latent_dim))
    for i in range(latent_dim):
        for j, space in enumerate(uniform_spacing):
            z_perturb = z.copy()
            z_perturb[i] = space
            z_interpolates[i, j] = z_perturb
    z_interpolates = z_interpolates.reshape((latent_dim * row, latent_dim))

    # decode
    with torch.no_grad():
        vae.decode.double()
        samples = vae.decode(torch.from_numpy(z_interpolates))

    # save image
    img_size = int(np.sqrt(opt.input_dim))

    save_image(
        samples.view(latent_dim * row, 1, img_size, img_size),
        f"./in_out/storage/results/{opt.experiment}_{opt.n_instances}/latent_walk_{interval}.png",
        nrow=latent_dim,
    )


if __name__ == "__main__":
    opt = parser.parse_args()

    path = "./in_out/storage/models/vae_dsprites_latent_2.pt"
    # model data contains VAE model, parameter mu and std, and losses
    model_data_2 = torch.load(
        "./in_out/storage/models/vae_dsprites_latent_2.pt",
        map_location=torch.device("cpu"),
    )
    model_data_8 = torch.load(
        "./in_out/storage/models/vae_dsprites_latent_8.pt",
        map_location=torch.device("cpu"),
    )
    model_data_32 = torch.load(
        "./in_out/storage/models/vae_dsprites_latent_32.pt",
        map_location=torch.device("cpu"),
    )
    model_data_128 = torch.load(
        "./in_out/storage/models/vae_dsprites_latent_128.pt",
        map_location=torch.device("cpu"),
    )

    loss = [
        model_data_2["train_loss"],
        model_data_8["train_loss"],
        model_data_32["train_loss"],
        model_data_128["train_loss"],
    ]

    plot_loss(loss)

    # walk_latent_space(row=8, interval=10)

    # explore_latent_variables(interval=10)
