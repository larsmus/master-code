import torch.nn.functional as F
import torch
import math


def get_loss(x, x_reconstructed, mu, logvar, opt, z, discriminator, count=1):
    if opt.model == "vae":
        return _vae_objective(x, x_reconstructed, mu, logvar, opt, count)
    elif opt.model == "b_vae_2":
        return _b_vae_objective2(x, x_reconstructed, mu, logvar, opt, count)
    elif opt.model == "factor_vae":
        return _factor_vae_objective(
            x, x_reconstructed, mu, logvar, opt, discriminator, z, count
        )
    elif opt.model == "btc_vae":
        return _btc_vae_objective(x, x_reconstructed, mu, logvar, opt, z, count)
    elif opt.model == "dip_vae":
        return _dip_vae_objective(x, x_reconstructed, mu, logvar, opt, count)
    else:
        raise NotImplementedError("Unknown loss function")


def _dip_vae_objective(x, x_reconstructed, mu, logvar, opt, count):
    vae_objective, reconstruction_error, kl_divergence, _ = _vae_objective(
        x, x_reconstructed, mu, logvar, opt, count
    )

    centered_mu = mu - mu.mean(dim=1, keepdim=True)
    cov_mu = centered_mu.t().matmul(centered_mu).squeeze()

    cov_z = cov_mu + torch.mean(torch.diagonal((2.0 * logvar).exp(), dim1=0), dim=0)
    cov_diag = torch.diag(cov_z)
    cov_off_diag = cov_z - torch.diag(cov_diag)

    dip_loss = opt.lambda_offdiag * torch.sum(
        cov_off_diag ** 2
    ) + opt.lambda_diag * torch.sum((cov_diag - 1) ** 2)

    return vae_objective + dip_loss, reconstruction_error, kl_divergence, dip_loss


def reconstruction(x, x_reconstructed, opt, distribution="bernoulli"):
    batch_size = x.size(0)
    if distribution == "bernoulli":
        return (
                F.binary_cross_entropy(
                    x_reconstructed, x.view(-1, opt.input_dim), reduction="sum"
                )
                / batch_size
        )

    elif distribution == "gaussian":
        return (
                       F.mse_loss(
                           x_reconstructed * 255, x.view(-1, opt.input_dim) * 255, reduction="sum"
                       )
                       / 255
               ) / batch_size

    else:
        raise ValueError("Unknown distribution")


def _kl_divergence(mu, logvar):
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kld.sum(1).mean(0, True)


def dimension_kld(mu, logvar):
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kld.sum(0)


def _vae_objective(x, x_reconstructed, mu, logvar, opt, count):
    annealing_c = _linear_annealing(0, 1, count, opt.annealing_steps)
    reconstruction_error = reconstruction(x, x_reconstructed, opt)
    kl_divergence = _kl_divergence(mu, logvar)
    return (
        reconstruction_error + annealing_c * opt.beta_regularizer * kl_divergence,
        reconstruction_error,
        kl_divergence,
        torch.tensor(0),
    )


def _b_vae_objective2(x, x_reconstructed, mu, logvar, opt, count):
    annealing_c = _linear_annealing(
        opt.C_initial, opt.C_final, count, opt.annealing_steps
    )
    reconstruction_error = reconstruction(x, x_reconstructed, opt)
    kl_divergence = _kl_divergence(mu, logvar)
    return (
        reconstruction_error + opt.gamma * (kl_divergence - annealing_c).abs(),
        reconstruction_error,
        kl_divergence,
        torch.tensor(0),
    )


def _factor_vae_objective(x, x_reconstructed, mu, logvar, opt, discriminator, z, count):
    log_probability = discriminator(z)
    total_correlation = (log_probability[:, :1] - log_probability[:, 1:]).mean()
    vae_objective, reconstruction_error, kl_divergence, _ = _vae_objective(
        x, x_reconstructed, mu, logvar, opt, count
    )
    return (
        vae_objective + opt.factor_regularizer * total_correlation,
        reconstruction_error,
        kl_divergence,
        total_correlation,
    )


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
    return 0.5 * (
            F.cross_entropy(log_probability, zeros)
            + F.cross_entropy(log_probability_permute, ones)
    )


def _permute_dims(z):
    assert z.dim() == 2

    batch_size, _ = z.size()
    z_permute = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(batch_size).to(z.device)
        z_j_permute = z_j[perm]
        z_permute.append(z_j_permute)

    return torch.cat(z_permute, 1)


def _btc_vae_objective(x, x_reconstructed, mu, logvar, opt, z, count):
    vae_objective, reconstruction_error, kl_divergence, _ = _vae_objective(
        x, x_reconstructed, mu, logvar, opt, count
    )

    log_q, log_q_product = _get_tc_estimates(z, mu, logvar)
    total_correlation = (log_q - log_q_product).mean()

    return (
        vae_objective + opt.btc_regularizer * total_correlation,
        reconstruction_error,
        kl_divergence,
        total_correlation,
    )


def _get_tc_estimates(z, mu, logvar, is_mss=False):
    batch_size, latent_dim = z.shape

    mat_log_q = _matrix_log_density_normal(z, mu, logvar)

    if is_mss:
        # use stratification
        log_iw_mat = _log_importance_weight_matrix(batch_size, n_data).to(
            latent_sample.device
        )
        mat_log_q = mat_log_q + log_iw_mat.view(batch_size, batch_size, 1)

    log_q = torch.logsumexp(mat_log_q.sum(2), dim=1, keepdim=False)
    log_q_product = torch.logsumexp(mat_log_q, dim=1, keepdim=False).sum(1)

    return log_q, log_q_product


def _log_density_normal(z, mu, logvar):
    normalization = -0.5 * (math.log(2 * math.pi) + logvar)
    inv_var = torch.exp(-logvar)
    log_density = normalization - 0.5 * ((z - mu) ** 2 * inv_var)
    return log_density


def _matrix_log_density_normal(z, mu, logvar):
    batch_size, dim = z.shape
    z = z.view(batch_size, 1, dim)
    mu = mu.view(1, batch_size, dim)
    logvar = logvar.view(1, batch_size, dim)
    return _log_density_normal(z, mu, logvar)


def _log_importance_weight_matrix(batch_size, dataset_size):
    n = dataset_size
    m = batch_size - 1
    strat_weight = (n - m) / (n * m)
    w = torch.Tensor(batch_size, batch_size).fill_(1 / m)
    w.view(-1)[:: m + 1] = 1 / n
    w.view(-1)[1:: m + 1] = strat_weight
    w[m - 1, 0] = strat_weight
    return w.log()


def _linear_annealing(initial, final, step, annealing_steps=0):
    if annealing_steps == 0:
        return final
    assert final > initial
    delta = final - initial
    annealed = min(initial + delta * step / annealing_steps, final)
    return annealed
