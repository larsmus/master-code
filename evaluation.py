from sklearn import linear_model
import numpy as np
import os
# to run
import torch
import json
from vae import ConvVAE


"""
Implement reconstruction error
z-score from beta vae
total correlation
mutual information gap
axis alignment metric
"""


class Evaluator:
    def __init__(self, data_path, model, opt, batch_size):

        self.opt = opt
        self.model = model
        self.opt = opt
        self.batch_size = batch_size

        data = np.load(data_path, encoding='latin1', allow_pickle=True)
        self.imgs = data["imgs"]
        self.factor_sizes = np.array(data["metadata"][()]["latents_sizes"], dtype=np.int64)
        self.latent_factor_indices = list(range(6))
        self.full_factor_sizes = [1, 3, 6, 40, 32, 32]

    def get_beta_metric(self, n_train, n_test):
        train_points, train_labels = generate_training_batch_beta(self, n_train)

        model = linear_model.LogisticRegression()
        model.fit(train_points, train_labels)

        train_accuracy = model.score(train_points, train_labels)
        # train_accuracy = np.mean(model.predict(train_points) == train_labels)

        eval_points, eval_labels = generate_training_batch_beta(self, n_test)

        eval_accuracy = model.score(eval_points, eval_labels)

        return train_accuracy, eval_accuracy

    def get_factor_metric(self, n_train, n_test, num_variance_estimate):
        pass

    def get_mig_metric(self):
        pass


def generate_training_batch_beta(evaluator, num_points):
    points = None  # Dimensionality depends on the representation function.
    labels = np.zeros(num_points, dtype=np.int64)
    for i in range(num_points):
        labels[i], feature_vector = generate_training_sample_beta(evaluator)
        if points is None:
            points = np.zeros((num_points, feature_vector.shape[0]))
        points[i, :] = feature_vector
    return points, labels


def generate_training_sample_beta(evaluator):
    num_factors = len(evaluator.latent_factor_indices)
    index = np.random.randint(num_factors)

    factors1 = _sample_factors(evaluator.latent_factor_indices, evaluator.factor_sizes, evaluator.batch_size)
    factors2 = _sample_factors(evaluator.latent_factor_indices, evaluator.factor_sizes, evaluator.batch_size)

    # Ensure sampled coordinate is the same across pairs of samples.
    factors2[:, index] = factors1[:, index]

    # Transform latent variables to observation space.
    observation1 = _sample_observations_from_factors(factors1, evaluator.latent_factor_indices, evaluator.factor_sizes,
                                                     evaluator.imgs)
    observation1 = torch.from_numpy(observation1).float().view(64, 1, 64, 64)
    observation2 = _sample_observations_from_factors(factors2, evaluator.latent_factor_indices, evaluator.factor_sizes,
                                                     evaluator.imgs)
    observation2 = torch.from_numpy(observation2).float().view(64, 1, 64, 64)

    # Compute representations based on the observations.
    with torch.no_grad():
        representation1, _ = evaluator.model.encoder(observation1)
        representation2, _ = evaluator.model.encoder(observation2)

    # Compute the feature vector based on differences in representation.
    feature_vector = np.mean(np.abs(representation1.numpy() - representation2.numpy()), axis=0)
    return index, feature_vector


def _sample_factors(latent_factor_indices, factor_sizes, num):
    factors = np.zeros(shape=(num, len(latent_factor_indices)), dtype=np.int64)
    for pos, i in enumerate(latent_factor_indices):
        factors[:, pos] = np.random.randint(factor_sizes[i], size=num)
    return factors


def _sample_observations_from_factors(factors, latent_factor_indices, factor_sizes, images):
    num_samples = factors.shape[0]
    num_factors = len(latent_factor_indices)

    all_factors = np.zeros(shape=(num_samples, num_factors), dtype=np.int64)
    all_factors[:, latent_factor_indices] = factors

    # Complete all the other factors
    observation_factor_indices = [i for i in range(num_factors) if i not in latent_factor_indices]

    for i in observation_factor_indices:
        all_factors[:, i] = np.random.randint(factor_sizes[i], size=num_samples)

    factor_bases = np.prod(factor_sizes) / np.cumprod(factor_sizes)
    indices = np.array(np.dot(all_factors, factor_bases), dtype=np.int64)

    return np.expand_dims(images[indices].astype(np.float32), axis=3)


if __name__ == "__main__":

    # run this file from the script folder
    data_path = "../data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

    n_train = 10000
    n_test = 5000

    path = "../results/dsprites/"
    models = ["vae", "factor_vae", "btc_vae", "dip_vae"]

    beta_metrics = {}

    for m in models:
        beta_metrics[m] = {}
        parameters = [p.split("parameter_")[1] for p in os.listdir(path + m + "/") if not p.startswith(".")]

        for p in parameters:
            beta_metrics[m][p] = {"train": [], "test": []}

            for i in range(10):
                model_path = path + m + f"/parameter_{p}/seed_{i+1}"
                model_data = torch.load(model_path + "/model.pt", map_location=torch.device('cpu'))
                model = ConvVAE(model_data["opt"])
                model.load_state_dict(state_dict=model_data["model"])
                opt = model_data["opt"]

                evaluate = Evaluator(data_path=data_path, model=model, opt=opt, batch_size=64)

                beta_metric_train, beta_metric_test = evaluate.get_beta_metric(n_train, n_test)
                beta_metrics[m][p]["train"].append(beta_metric_train)
                beta_metrics[m][p]["test"].append(beta_metric_test)

    with open(path + "beta_metrics.json", "w") as f:
        json.dump(beta_metrics, f)
