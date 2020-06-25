import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import View
import numpy as np


# Simple VAE model with fully connected MLP structure with one hidden layer


class Vae(nn.Module):
    def __init__(self, opt):
        super(Vae, self).__init__()

        self.encode = nn.Sequential(
            nn.Linear(opt.input_dim, opt.hidden_dim),
            nn.Linear(opt.hidden_dim, opt.hidden_dim),
            nn.Linear(opt.hidden_dim, 2 * opt.latent_dim),
        )

        self.decode = nn.Sequential(
            nn.Linear(opt.latent_dim, opt.hidden_dim),
            nn.Linear(opt.hidden_dim, opt.hidden_dim),
            nn.Linear(opt.hidden_dim, opt.input_dim),
        )

        self.input_dim = opt.input_dim
        self.mu = 0
        self.std = 0
        self.weight_init()

    def _encode(self, x):
        # compute the mean and standard deviation each size 'batch size x z_dim'
        # compute fist log and then take the exponential for std to ensure that it is positive
        encoded = self.encode(x)
        z_loc = encoded[:, : self.opt.latent_dim]
        z_scale = encoded[:, self.opt.latent_dim :]
        return z_loc, z_scale

    def _decode(self, x):
        # ensure output in [0,1] domain
        # return torch.sigmoid(self.decode(x))
        return self.decode(x)

    def forward(self, x):
        mu, log_std = self._encode(x)
        z = reparameterize(mu=mu, logvar=log_std)
        self.mu = mu
        self.std = log_std
        return self._decode(z), mu, log_std

    def sample(self, mu, std):
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std
        return self.decode(z)

    def loss(self, x, x_reconstructed, mu, logvar):
        # Compute ELBO. For normal prior this has closed form.
        # Can use BCE sine we use sigmoid in decoder to output Bernoulli probabilities
        reconstruction = F.binary_cross_entropy(
            x_reconstructed, x.view(-1, self.input_dim), reduction="sum"
        )
        # KL term regularizing between variational distribution and prior. Using closed form.
        # See https://arxiv.org/pdf/1312.6114.pdf Appendix B and C for details.
        KL_regularizer = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction + KL_regularizer


# DCGAN like architecture
class ConvVAE(nn.Module):
    def __init__(self, opt):
        super(ConvVAE, self).__init__()

        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        # Shape required to start transpose convs
        reshape = (hid_channels, kernel_size, kernel_size)
        cnn_kwargs = dict(stride=2, padding=1)

        self.encode = nn.Sequential(
            nn.Conv2d(
                in_channels=opt.channels,
                out_channels=hid_channels,
                kernel_size=kernel_size,
                **cnn_kwargs
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hid_channels,
                out_channels=hid_channels,
                kernel_size=kernel_size,
                **cnn_kwargs
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hid_channels,
                out_channels=hid_channels,
                kernel_size=kernel_size,
                **cnn_kwargs
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hid_channels,
                out_channels=hid_channels,
                kernel_size=kernel_size,
                **cnn_kwargs
            ),
            nn.ReLU(inplace=True),
            View((-1, np.product(reshape))),
            nn.Linear(in_features=np.product(reshape), out_features=hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_dim, out_features=opt.latent_dim * 2),
        )

        self.decode = nn.Sequential(
            nn.Linear(in_features=opt.latent_dim, out_features=hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_dim, out_features=np.product(reshape)),
            nn.ReLU(inplace=True),
            View((-1, *reshape)),
            nn.ConvTranspose2d(
                in_channels=hid_channels,
                out_channels=hid_channels,
                kernel_size=kernel_size,
                **cnn_kwargs
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=hid_channels,
                out_channels=hid_channels,
                kernel_size=kernel_size,
                **cnn_kwargs
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=hid_channels,
                out_channels=hid_channels,
                kernel_size=kernel_size,
                **cnn_kwargs
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=hid_channels,
                out_channels=opt.channels,
                kernel_size=kernel_size,
                **cnn_kwargs
            ),
        )

        self.opt = opt
        self.mu = 0
        self.logvar = 0
        self.current_training_iteration = 0

    def encoder(self, x):
        # compute the mean and standard deviation each size 'batch size x z_dim'
        # compute fist log and then take the exponential for std to ensure that it is positive
        encoded = self.encode(x)
        z_loc = encoded[:, : self.opt.latent_dim]
        z_scale = encoded[:, self.opt.latent_dim:]
        return z_loc, z_scale

    def decoder(self, x):
        # ensure output in [0,1] domain
        return torch.sigmoid(self.decode(x))

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu=mu, logvar=logvar)
        self.mu = mu
        self.logvar = logvar
        return self.decoder(z), mu, logvar, z

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std
        return self.decoder(z)


class Discriminator(nn.Module):
    def __init__(self, z_dim):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 2),
        )

    def forward(self, z):
        return self.net(z).squeeze()


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    epsilon = torch.randn_like(std)
    return mu + epsilon * std
