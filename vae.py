import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from src.utils import View


# Simple VAE model with fully connected MLP structure with one hidden layer


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.input_dim = opt.input_dim
        # define architecture
        self.fc1 = nn.Linear(opt.input_dim, opt.hidden_dim)
        self.fc2 = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.fc21 = nn.Linear(opt.hidden_dim, opt.latent_dim)
        self.fc22 = nn.Linear(opt.hidden_dim, opt.latent_dim)

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        hidden = F.relu(self.fc1(x))
        hidden = F.relu(self.fc2(hidden))
        # compute the mean and standard deviation each size 'batch size x z_dim'
        z_loc = self.fc21(hidden)
        z_scale = self.fc22(hidden)
        return z_loc, z_scale


class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()

        # define architecture
        self.fc1 = nn.Linear(opt.latent_dim, opt.hidden_dim)
        self.fc2 = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.fc3 = nn.Linear(opt.hidden_dim, opt.input_dim)

    def forward(self, x):
        hidden = F.relu(self.fc1(x))
        hidden = F.relu(self.fc2(hidden))
        # ensure output in [0,1] domain
        return torch.sigmoid(self.fc3(hidden))


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

        self.encode = nn.Sequential(
            nn.Conv2d(
                in_channels=opt.channels,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
            View((-1, 256 * 1 * 1)),
            nn.Linear(in_features=256, out_features=opt.latent_dim * 2),
        )

        self.decode = nn.Sequential(
            nn.Linear(in_features=opt.latent_dim, out_features=256),
            View((-1, 256, 1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=opt.channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )

        self.opt = opt
        self.mu = 0
        self.std = 0
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

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

    def loss(self, x, x_reconstructed, mu, log_std, distribution="gaussian"):
        # Compute ELBO.
        if distribution == "bernoulli":
            # Can use BCE sine we use sigmoid in decoder to output Bernoulli probabilities
            reconstruction = F.binary_cross_entropy(
                x_reconstructed, x.view(-1, self.opt.input_dim), size_average=False
            )
        elif distribution == "gaussian":
            reconstruction = F.mse_loss(
                x_reconstructed, x.view(-1, self.opt.input_dim), size_average=False
            )

        # KL term regularizing between variational distribution and prior. Using closed form.
        # See https://arxiv.org/pdf/1312.6114.pdf Appendix B and C for details.
        KL_regularizer = -0.5 * torch.sum(1 + log_std - mu.pow(2) - log_std.exp())
        loss = reconstruction + self.opt.beta_regularizer * KL_regularizer
        return loss


def reparameterize(mu, log_std):
    std = torch.exp(0.5 * log_std)
    epsilon = torch.randn_like(std)
    return mu + epsilon * std


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()
