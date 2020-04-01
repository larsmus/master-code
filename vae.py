import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
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
        z = reparameterize(mu=mu, log_std=log_std)
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

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)


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
            )
        )

        self.opt = opt
        self.mu = 0
        self.std = 0

    def _encode(self, x):
        # compute the mean and standard deviation each size 'batch size x z_dim'
        # compute fist log and then take the exponential for std to ensure that it is positive
        encoded = self.encode(x)
        z_loc = encoded[:, : self.opt.latent_dim]
        z_scale = encoded[:, self.opt.latent_dim :]
        return z_loc, z_scale

    def _decode(self, x):
        # ensure output in [0,1] domain
        return torch.sigmoid(self.decode(x))

    def forward(self, x):
        mu, log_std = self._encode(x)
        z = reparameterize(mu=mu, log_std=log_std)
        self.mu = mu
        self.std = log_std
        return self._decode(z), mu, log_std

    def sample(self, mu, std):
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std
        return self.decode(z)

    def loss(self, x, x_reconstructed, mu, log_std, distribution="bernoulli"):
        # Compute ELBO.
        batch_size = x.size(0)
        if distribution == "bernoulli":
            # Can use BCE sine we use sigmoid in decoder to output Bernoulli probabilities
            reconstruction = F.binary_cross_entropy(
                x_reconstructed, x.view(-1, self.opt.input_dim), reduction="sum"
            )

        elif distribution == "gaussian":
            reconstruction = (
                F.mse_loss(
                    x_reconstructed * 255,
                    x.view(-1, self.opt.input_dim) * 255,
                    reduction="sum",
                )
                / 255
            )

        else:
            raise ValueError("Unknown distribution")

        reconstruction = reconstruction / batch_size

        # KL term regularizing between variational distribution and prior. Using closed form.
        # See https://arxiv.org/pdf/1312.6114.pdf Appendix B and C for details.
        kld = -0.5 * (1 + log_std - mu.pow(2) - log_std.exp())
        total_kld = kld.sum(1).mean(0, True)
        dim_kld = kld.sum(0)
        loss = reconstruction + self.opt.beta_regularizer * total_kld
        return loss, dim_kld


def reparameterize(mu, log_std):
    std = torch.exp(0.5 * log_std)
    epsilon = torch.randn_like(std)
    return mu + epsilon * std

