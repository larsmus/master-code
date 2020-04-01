import torch
from torch import optim
import argparse
import os
from load_data import get_dataloader
from vae import Vae, ConvVAE, kaiming_init
import time
from datetime import datetime
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
    parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
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
    parser.add_argument(
        "--test", type=int, default=0, help="Use test set or not. 1 or 0"
    )
    parser.add_argument(
        "--store-samples",
        type=int,
        default=0,
        help="If store samples during training. 1 or 0",
    )
    return parser.parse_args()


def train(dataloader):
    vae.train()
    train_loss = 0
    dimension_kld_sum = torch.zeros(10, device=device)
    for batch_idx, data in enumerate(dataloader):
        optimizer.zero_grad()
        if torch.cuda.is_available():
            data = data.cuda()
        data = data.view(opt.batch_size, opt.channels, opt.resolution, opt.resolution)
        reconstruction, mu, std = vae(data)
        loss_value, dim_kld_batch = vae.loss(data, reconstruction, mu, std)
        dimension_kld_sum += dim_kld_batch
        loss_value.backward()
        train_loss += loss_value.item()
        optimizer.step()

    return train_loss, mu, std, dimension_kld_sum


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


def run():

    for epoch in range(opt.n_epoch):
        train_loss, mu, std, dimension_kld_sum = train(train_dataloader)
        dimension_kld = dimension_kld_sum / n
        if bool(opt.test):
            losses_test.append(test(test_dataloader))
        losses_train.append(train_loss / n)
        print(
            f"[Epoch {epoch + 1:04}/{opt.n_epoch:04}] [Train loss: {train_loss / n:.2f}]"
        )
        if opt.store_samples:
            with torch.no_grad():
                sample, dim = vae.sample(mu, std)
                os.makedirs(out_path + "/samples", exist_ok=True)
                batch_size = sample.shape[0]
                save_image(
                    sample.view(
                        batch_size, opt.channels, opt.resolution, opt.resolution
                    ),
                    out_path + "/samples/" + str(epoch) + ".png",
                )

    return dimension_kld


if __name__ == "__main__":

    opt = parse()
    torch.manual_seed(opt.seed)

    os.makedirs(f"../results/{opt.dataset}", exist_ok=True)
    # run_id = datetime.now().strftime("%d-%m-%Y,%H-%M-%S")
    # out_path = f"../results/{opt.dataset}/{run_id}"
    out_path = f"../results/{opt.dataset}/betas_lr/beta_{opt.beta_regularizer}"
    os.makedirs(out_path, exist_ok=True)

    # check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    print("Loading data")
    train_dataloader = get_dataloader(opt)
    test_dataloader = None
    n = len(train_dataloader.dataset)

    # run
    start = time.time()
    print("Training")
    vae = ConvVAE(opt).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    losses_train = []
    losses_test = []
    dimension_kld = run()
    print(f"Done! Total time training: {time.time() - start:.1f} seconds")

    # store loss and model
    torch.save(
        {
            "model": vae.state_dict(),
            "mu": vae.mu,
            "std": vae.std,
            "train_loss": losses_train,
            "opt": opt,
            "dim_kld": dimension_kld,
        },
        out_path + "/model.pt",
    )
