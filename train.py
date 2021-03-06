import torch
from torch import optim
import argparse
import os
from load_data import get_dataloader
from vae import ConvVAE, Discriminator
import time
from datetime import datetime
from torchvision.utils import save_image
import warnings
from objectives import get_loss, dimension_kld, factor_vae_discriminator_loss
import math

warnings.filterwarnings("ignore")


def parse():
    parser = argparse.ArgumentParser(description="VAE")

    general = parser.add_argument_group("General options")
    general.add_argument(
        "--dataset", type=str, default="dsprites", help="which dataset to run"
    )
    general.add_argument(
        "--log_interval", type=int, default=10, help="Interval to log loss"
    )
    general.add_argument("--seed", type=int, default=123, help="Seed")
    general.add_argument(
        "--test", type=int, default=0, help="Use test set or not. 1 or 0"
    )
    general.add_argument(
        "--store-samples",
        type=int,
        default=0,
        help="If store samples during training. 1 or 0",
    )

    training = parser.add_argument_group("Training options")
    training.add_argument(
        "--batch_size", type=int, default=64, help="number of datapoints in each batch"
    )
    training.add_argument("--n_epoch", type=int, default=100, help="number of epochs")
    training.add_argument("--lr", type=float, default=0.0005, help="learning rate")
    training.add_argument("--b1", type=float, default=0.9, help="parameter in Adam")
    training.add_argument("--b2", type=float, default=0.999, help="parameter in Adam")

    training.add_argument("--lr_d", type=float, default=0.0001, help="learning rate")
    training.add_argument("--b1_d", type=float, default=0.5, help="parameter in Adam")
    training.add_argument("--b2_d", type=float, default=0.9, help="parameter in Adam")

    model = parser.add_argument_group("Model options")
    model.add_argument("--model", type=str, default="vae", help="which model to run")
    model.add_argument("--resolution", type=int, default=64, help="resolution of image")
    model.add_argument(
        "--input_dim", type=int, default=64 * 64, help="dimension of input space"
    )
    model.add_argument(
        "--latent_dim", type=int, default=10, help="dimension of latent space"
    )
    model.add_argument("--channels", type=int, default=1, help="Number of channels")
    model.add_argument(
        "--loss", type=str, default="beta_vae_2", help="Which loss function to use"
    )

    beta_vae_1 = parser.add_argument_group("Loss options for beta-vae, first version")
    beta_vae_1.add_argument(
        "--beta_regularizer", type=float, default=5, help="beta in beta-VAE"
    )
    beta_vae_1.add_argument(
        "--annealing_steps", type=int, default=0, help="Use annealing on beta or not"
    )
    beta_vae_1.add_argument(
        "--beta_annealing", type=int, default=0, help="Use annealing on beta or not"
    )

    beta_vae_2 = parser.add_argument_group("Loss options for beta-vae, second version")
    beta_vae_2.add_argument(
        "--gamma", type=int, default=100, help="Regularizer on KL term"
    )
    beta_vae_2.add_argument("--C_initial", type=int, default=0, help="Initial capacity")
    beta_vae_2.add_argument("--C_final", type=int, default=25, help="Final capacity")

    factor_vae = parser.add_argument_group("Loss options for FactorVAE")
    factor_vae.add_argument(
        "--factor_regularizer", type=int, default=5, help="Regularizer on TC term"
    )
    factor_vae.add_argument("--lrd", type=float, default=0.0001, help="learning rate")
    factor_vae.add_argument("--b1d", type=float, default=0.5, help="parameter in Adam")
    factor_vae.add_argument("--b2d", type=float, default=0.9, help="parameter in Adam")

    btc_vae = parser.add_argument_group("Loss options for bet-tc-vae")
    btc_vae.add_argument(
        "--btc_regularizer", type=int, default=5, help="Regularizer on TC term"
    )

    dip_vae = parser.add_argument_group("Loss options for dip-vae")
    dip_vae.add_argument(
        "--lambda_diag", type=int, default=5, help="Regularizer on diagonal term"
    )
    dip_vae.add_argument(
        "--lambda_offdiag", type=int, default=5, help="Regularizer on off-diagonal term"
    )
    return parser.parse_args()


def train(dataloader, epoch):
    vae.train()
    train_loss = 0
    recon_epoch = 0
    kl_epoch = 0
    tc_epoch = 0

    mu = None
    logvar = None
    dimension_kld_sum = torch.zeros(10, device=device)

    for batch_idx, data in enumerate(dataloader):
        optimizer.zero_grad()
        data = data.to(device)

        data = data.view(opt.batch_size, opt.channels, opt.resolution, opt.resolution)
        x_reconstructed, mu, logvar, z = vae(data)

        loss_value, recon, kl, tc = get_loss(
            data,
            x_reconstructed,
            mu,
            logvar,
            opt,
            z,
            discriminator,
            count=epoch + batch_idx,
        )

        recon_epoch += recon.item()
        kl_epoch += kl.item()
        tc_epoch += tc.item()

        if opt.model == "factor_vae":
            loss_value.backward(retain_graph=True)
            train_loss += loss_value.item()

            optimizer_d.zero_grad()
            d_loss = factor_vae_discriminator_loss(
                z, discriminator, dataloader, vae, opt, device
            )
            discriminator_loss.append(d_loss.item())
            d_loss.backward()
            optimizer_d.step()

        else:
            loss_value.backward()
            train_loss += loss_value.item()

        if epoch == (opt.n_epoch - 1):
            dimension_kld_batch = dimension_kld(mu, logvar)
            dimension_kld_sum += dimension_kld_batch

        optimizer.step()

    return train_loss, mu, logvar, dimension_kld_sum, recon_epoch, kl_epoch, tc_epoch


def test(dataloader):
    vae.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)
            x_reconstructed, mu, logvar = vae(data)
            loss_value = vae.loss(data, x_reconstructed, mu, logvar)
            test_loss += loss_value.item()
    return test_loss


def run():

    dimension_kld_sum = None

    for epoch in range(opt.n_epoch):
        train_loss, mu, logvar, dimension_kld_sum, recon_epoch, kl_epoch, tc_epoch = train(
            train_dataloader, epoch
        )

        if bool(opt.test):
            losses_test.append(test(test_dataloader))
        losses_train.append(train_loss / n)
        reconstruction_error.append(recon_epoch / n)
        kl_divergence.append(kl_epoch / n)
        total_correlation.append(tc_epoch / n)

        print(
            f"[Epoch {epoch + 1:04}/{opt.n_epoch:04}] [Train loss: {train_loss / n:.2f}]"
        )

        if opt.store_samples:
            with torch.no_grad():
                sample, dim = vae.sample(mu, logvar)
                os.makedirs(out_path + "/samples", exist_ok=True)
                batch_size = sample.shape[0]
                save_image(
                    sample.view(
                        batch_size, opt.channels, opt.resolution, opt.resolution
                    ),
                    out_path + "/samples/" + str(epoch) + ".png",
                )

    return dimension_kld_sum / n


if __name__ == "__main__":

    opt = parse()
    torch.manual_seed(opt.seed)

    os.makedirs(f"../results/{opt.dataset}", exist_ok=True)
    # run_id = datetime.now().strftime("%d-%m-%Y,%H-%M-%S")
    # out_path = f"../results/{opt.dataset}/{run_id}"

    if opt.model == "factor_vae":
        parameter = opt.factor_regularizer
    elif opt.model == "vae":
        parameter = opt.beta_regularizer
    elif opt.model == "btc_vae":
        parameter = opt.btc_regularizer
    elif opt.model == "dip_vae":
        parameter = opt.lambda_diag
    else:
        parameter = 0

    out_path = (
        f"../results/{opt.dataset}/{opt.model}/parameter_{parameter}/seed_{opt.seed}"
    )
    os.makedirs(out_path, exist_ok=True)

    # check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    print("Loading data")
    train_dataloader = get_dataloader(opt)
    test_dataloader = None
    n = len(train_dataloader.dataset)
    iter_per_epoch = math.ceil(n / opt.batch_size)
    # run
    start = time.time()
    print("Training")

    vae = ConvVAE(opt).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    discriminator = Discriminator(opt.latent_dim).to(device)
    optimizer_d = optim.Adam(
        discriminator.parameters(), lr=opt.lrd, betas=(opt.b1d, opt.b2d)
    )

    losses_train = []
    reconstruction_error = []
    kl_divergence = []
    total_correlation = []
    discriminator_loss = []
    losses_test = []

    dimension_kld = run()
    print(f"Done! Total time training: {time.time() - start:.1f} seconds")

    # store loss and model
    torch.save(
        {
            "model": vae.state_dict(),
            "mu": vae.mu,
            "logvar": vae.logvar,
            "train_loss": losses_train,
            "opt": opt,
            "dim_kld": dimension_kld,
            "reconstruction_error": reconstruction_error,
            "kl_divergence": kl_divergence,
            "total_correlation": total_correlation,
            "discriminator_loss:": discriminator_loss,
        },
        out_path + "/model.pt",
    )
