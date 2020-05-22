import torch.nn.functional as F
import torch


def get_loss(x, x_reconstructed, mu, log_std, opt, z,  discriminator, count=1):
    if opt.model == "vae":
        return vae_objective(x, x_reconstructed, mu, log_std, opt, count)
    elif opt.model == "b_vae_2":
        return b_vae_objective2(x, x_reconstructed, mu, log_std, opt, count)
    elif opt.model == "factor_vae":
        return factor_vae_objective(x, x_reconstructed, mu, log_std, opt, discriminator, z, count)
    else:
        raise NotImplementedError("Unknown loss function")


def _reconstruction(x, x_reconstructed, opt, distribution="bernoulli"):
    batch_size = x.size(0)
    if distribution == "bernoulli":
        return F.binary_cross_entropy(
            x_reconstructed, x.view(-1, opt.input_dim), reduction="sum"
        ) / batch_size

    elif distribution == "gaussian":
        return (F.mse_loss(
                    x_reconstructed * 255,
                    x.view(-1, opt.input_dim) * 255,
                    reduction="sum",
                ) / 255) / batch_size

    else:
        raise ValueError("Unknown distribution")


def kl_divergence(mu, log_std):
    kld = -0.5 * (1 + log_std - mu.pow(2) - log_std.exp())
    return kld.sum(1).mean(0, True)


def dimension_kld(mu, log_std):
    kld = -0.5 * (1 + log_std - mu.pow(2) - log_std.exp())
    return kld.sum(0)


def vae_objective(x, x_reconstructed, mu, log_std, opt, count):
    annealing_c = linear_annealing(opt.C_initial, opt.C_final, count, opt.annealing_steps)
    return _reconstruction(x, x_reconstructed, opt) + annealing_c * opt.beta_regularizer * kl_divergence(mu, log_std)


def b_vae_objective2(x, x_reconstructed, mu, log_std, opt, count):
    annealing_c = linear_annealing(opt.C_initial, opt.C_final, count, opt.annealing_steps)
    return _reconstruction(x, x_reconstructed, opt) + opt.gamma * (kl_divergence(mu, log_std) - annealing_c).abs()


def factor_vae_objective(x, x_reconstructed, mu, log_std, opt, discriminator, z, count):
    log_probability = discriminator(z)
    total_correlation = (log_probability[:, :1] - log_probability[:, 1:]).mean()
    return vae_objective(x, x_reconstructed, mu, log_std, opt, count) + opt.factor_regularizer * total_correlation


def factor_vae_discriminator_loss(z, discriminator, dataloader, vae, opt, device):
    ones = torch.ones(opt.batch_size, dtype=torch.long, device=device)
    zeros = torch.zeros_like(ones)

    x_new = next(iter(dataloader))
    x_new = x_new.view(opt.batch_size, opt.channels, opt.resolution, opt.resolution)
    x_new = x_new.to(device)

    _, _, _, z_new = vae(x_new)
    z_permute = _permute_dims(z_new)

    log_probability_permute = discriminator(z_permute)
    log_probability = discriminator(z)
    return 0.5 * (F.cross_entropy(log_probability, zeros) + F.cross_entropy(log_probability_permute, ones))


def _permute_dims(z):
    assert z.dim() == 2

    batch_size, _ = z.size()
    z_permute = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(batch_size).to(z.device)
        z_j_permute = z_j[perm]
        z_permute.append(z_j_permute)

    return torch.cat(z_permute, 1)


def btc_vae_objective(x, x_reconstructed, mu, log_std, opt, z, count):
    elbo = vae_objective(x, x_reconstructed, mu, log_std, opt, count)

    log_q, log_q_product = _get_tc_estimates(z, mu, log_std)
    total_correlation = (log_q - log_q_product).mean()

    return elbo + total_correlation


def _get_tc_estimates(z, mu, log_std):
    batch_size, latent_dim = z.shape


    return log_g, log_q_product


def linear_annealing(initial, final, step, annealing_steps=0):
    if annealing_steps == 0:
        return final
    assert final > initial
    delta = final - initial
    annealed = min(initial + delta * step / annealing_steps, final)
    return annealed


