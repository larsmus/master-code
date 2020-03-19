import torch
from torch import optim
import argparse
import os
from src.load_data import get_dataloader
from src.vae import Vae, ConvVAE
import time
from torchvision.utils import save_image
import warnings

warnings.filterwarnings("ignore")


def parse():
    parser = argparse.ArgumentParser(description="VAE")
    parser.add_argument(
        "--dataset", type=str, default="dsprites", help="which dataset to run"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="triangles",
        help="which experiment if polygons",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="number of datapoints in each batch"
    )
    parser.add_argument("--n_epoch", type=int, default=100, help="number of epochs")
    parser.add_argument(
        "--resolution", type=int, default=64, help="resolution of image"
    )
    parser.add_argument(
        "--input_dim", type=int, default=64 * 64, help="dimension of input space"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=10, help="dimension of latent space"
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="parameter in Adam")
    parser.add_argument("--b2", type=float, default=0.999, help="parameter in Adam")
    parser.add_argument(
        "--log_interval", type=int, default=10, help="Interval to log loss"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=400, help="hidden units in 3 layer MLP"
    )
    parser.add_argument("--seed", type=int, default=123, help="Seed")
    parser.add_argument("--channels", type=str, default=1, help="Number of channels")
    parser.add_argument(
        "--beta_regularizer", type=float, default=1.0, help="Regularizer in beta-VAE"
    )
    return parser.parse_args()


def train(dataloader):
    vae.train()
    train_loss = 0
    for batch_idx, data in enumerate(dataloader):
        if torch.cuda.is_available():
            data = data.cuda()
        data = data.view(opt.batch_size, opt.channels, opt.resolution, opt.resolution)
        reconstruction, mu, std = vae(data)
        loss_value = vae.loss(data, reconstruction, mu, std)
        optimizer.zero_grad()
        loss_value.backward()
        train_loss += loss_value.item()
        optimizer.step()

    return train_loss, mu, std


def test(dataloader):
    vae.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            data.to(device)
            reconstruction, mu, std = vae(data)
            loss_value = vae.loss(data, reconstruction, mu, std)
            test_loss += loss_value.item()
    return test_loss


# run on both a train and a test dataset
def run_split():
    for epoch in range(opt.n_epoch):
        train_loss, mu, std = train(train_dataloader)
        test_loss = test(test_dataloader)
        losses_train.append(train_loss / data_options["n_instances"])
        losses_test.append(test_loss / data_options["n_test"])
        print(
            f"[Epoch {epoch + 1:04}/{opt.n_epoch:04}] [Test loss: {train_loss / data_options['n_instances']:.2f}] "
            f"[Test loss: {test_loss / data_options['n_test']:.2f}]"
        )
        with torch.no_grad():
            sample = vae.sample(mu, std)
            os.makedirs(
                f"./in_out/storage/results/{opt.experiment}_{data_options['n_instances']}",
                exist_ok=True,
            )
            save_image(
                sample.view(64, 1, opt.resolution, opt.resolution),
                f"./in_out/storage/results/{opt.experiment}_{data_options['n_instances']}/sample_"
                + str(epoch)
                + ".png",
            )


def run():
    for epoch in range(opt.n_epoch):
        train_loss, mu, std = train(train_dataloader)
        losses_train.append(train_loss / n)
        print(
            f"[Epoch {epoch + 1:04}/{opt.n_epoch:04}] [Train loss: {train_loss / n:.2f}]"
        )
        with torch.no_grad():
            sample = vae.sample(mu, std)
            os.makedirs(
                f"./in_out/storage/results/{opt.dataset}_latent_{opt.latent_dim}",
                exist_ok=True,
            )
            batch_size = sample.shape[0]
            save_image(
                sample.view(batch_size, opt.channels, opt.resolution, opt.resolution),
                f"./in_out/storage/results/{opt.dataset}_latent_{opt.latent_dim}/sample_"
                + str(epoch)
                + ".png",
            )


if __name__ == "__main__":

    opt = parse()
    torch.manual_seed(opt.seed)

    # check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    print("Loading data")
    train_dataloader, _ = get_dataloader(opt)
    n = len(train_dataloader.dataset)

    # run
    start = time.time()
    print("Training")
    vae = ConvVAE(opt).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    losses_train = []
    run()
    print(f"Done! Total time training: {time.time() - start:.1f} seconds")

    # store loss and model
    os.makedirs("./in_out/storage/models", exist_ok=True)
    torch.save(
        {
            "model": vae,
            "mu": vae.mu,
            "std": vae.std,
            "train_loss": losses_train,
            "opt": opt,
        },
        f"./in_out/storage/models/vae_{opt.dataset}_latent_{opt.latent_dim}.pt",
    )
