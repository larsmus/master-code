from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.metrics import mutual_info_score
import numpy as np
import os
import argparse

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

        data = np.load(data_path, encoding="latin1", allow_pickle=True)
        self.imgs = data["imgs"]
        self.factor_sizes = np.array(
            data["metadata"][()]["latents_sizes"], dtype=np.int64
        )
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
        global_variances = _compute_variances(self, num_variance_estimate)
        active_dims = _prune_dims(global_variances)

        training_votes = generate_training_batch_factor(
            self, n_train, global_variances, active_dims
        )

        classifier = np.argmax(training_votes, axis=0)
        other_index = np.arange(training_votes.shape[1])

        train_accuracy = (
                np.sum(training_votes[classifier, other_index])
                * 1.0
                / np.sum(training_votes)
        )

        eval_votes = generate_training_batch_factor(
            self, n_test, global_variances, active_dims
        )

        eval_accuracy = (
                np.sum(eval_votes[classifier, other_index]) * 1.0 / np.sum(eval_votes)
        )

        return train_accuracy, eval_accuracy

    def get_mig_metric(self, num_bins):
        n = len(self.imgs)
        representation, factors = generate_training_batch_mig(self, n, batch_size=16)

        assert representation.shape[1] == n

        return _compute_mig(representation, factors, num_bins)

    def get_downstream(self, num_train, num_test=5000):
        scores = {}
        for train_size in num_train:
            representation, factors = generate_training_batch_mig(
                self, train_size, batch_size=10
            )
            representation_test, factors_test = generate_training_batch_mig(
                self, num_test, batch_size=10
            )

            train_error, test_error = _compute_loss(
                np.transpose(representation),
                factors,
                np.transpose(representation_test),
                factors_test,
            )

            size_string = str(train_size)
            scores[size_string] = {}
            scores[size_string]["mean_train_accuracy"] = np.mean(train_error)
            scores[size_string]["mean_test_accuracy"] = np.mean(test_error)
            scores[size_string]["min_train_accuracy"] = np.min(train_error)
            scores[size_string]["min_test_accuracy"] = np.min(test_error)
            for i in range(len(train_error)):
                scores[size_string]["train_accuracy_factor_{}".format(i)] = train_error[
                    i
                ]
                scores[size_string]["test_accuracy_factor_{}".format(i)] = test_error[i]

        return scores


def _get_prediction_model(prediction_model):
    if prediction_model == "logistic":
        return linear_model.LogisticRegressionCV(Cs=10, cv=KFold(n_splits=5))
    elif prediction_model == "gbt":
        return GradientBoostingClassifier()
    else:
        raise NotImplementedError("Unknown prediction model")


def _compute_loss(x_train, y_train, x_test, y_test):
    """Compute average accuracy for train and test set."""
    num_factors = y_train.shape[0]
    train_loss = []
    test_loss = []
    for i in range(num_factors - 1):
        prediction_model = _get_prediction_model(args.prediction_model)
        prediction_model.fit(x_train, y_train[i + 1, :])
        train_loss.append(
            np.mean(prediction_model.predict(x_train) == y_train[i + 1, :])
        )
        test_loss.append(np.mean(prediction_model.predict(x_test) == y_test[i + 1, :]))
    return train_loss, test_loss


def generate_training_batch_mig(evaluator, n, batch_size):
    representations = None
    factors = None
    i = 0
    while i < n:
        num_points_iter = min(n - i, batch_size)
        current_factors = _sample_factors(
            evaluator.latent_factor_indices, evaluator.factor_sizes, num_points_iter
        )
        current_observations = _sample_observations_from_factors(
            current_factors,
            evaluator.latent_factor_indices,
            evaluator.factor_sizes,
            evaluator.imgs,
        )
        current_observations = (
            torch.from_numpy(current_observations)
                .float()
                .view(num_points_iter, 1, 64, 64)
                .to(device)
        )
        if i == 0:
            factors = current_factors
            with torch.no_grad():
                representations, _ = evaluator.model.encoder(current_observations)
                representations = representations.numpy()
        else:
            factors = np.vstack((factors, current_factors))
            with torch.no_grad():
                current_representations, _ = evaluator.model.encoder(
                    current_observations
                )
                current_representations = current_representations.numpy()
            representations = np.vstack((representations, current_representations))
        i += num_points_iter
    return np.transpose(representations), np.transpose(factors)


def _compute_mig(representation, factors, num_bins):
    discretized_representation = _discretize(representation, num_bins)
    mutual_info = _discrete_mutual_info(discretized_representation, factors)
    assert mutual_info.shape[0] == representation.shape[0]
    assert mutual_info.shape[1] == factors.shape[0]
    entropy = _discrete_entropy(factors)
    sorted_mutual_info = np.sort(mutual_info, axis=0)[::-1]
    return np.mean(
        np.divide(sorted_mutual_info[0, 1:] - sorted_mutual_info[1, 1:], entropy[1:])
    )


def _discrete_entropy(factors):
    num_factors = factors.shape[0]
    h = np.zeros(num_factors)
    for j in range(num_factors):
        h[j] = mutual_info_score(factors[j, :], factors[j, :])
    return h


def _discrete_mutual_info(representation, factors):
    num_codes = representation.shape[0]
    num_factors = factors.shape[0]
    m = np.zeros([num_codes, num_factors])
    for i in range(num_codes):
        for j in range(num_factors):
            m[i, j] = mutual_info_score(factors[j, :], representation[i, :])
    return m


def _discretize(target, num_bins):
    """Discretization based on histograms."""
    discretized = np.zeros_like(target)
    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(
            target[i, :], np.histogram(target[i, :], num_bins)[1][:-1]
        )
    return discretized


def generate_training_batch_factor(
        evaluator, num_points, global_variances, active_dims
):
    num_factors = len(evaluator.latent_factor_indices)
    votes = np.zeros((num_factors, global_variances.shape[0]), dtype=np.int64)
    for _ in range(num_points):
        factor_index, arg_min = generate_training_sample_factor(
            evaluator, global_variances, active_dims
        )
        votes[factor_index, arg_min] += 1
    return votes


def generate_training_sample_factor(evaluator, global_variances, active_dims):
    # Select random coordinate to keep fixed.
    num_factors = len(evaluator.latent_factor_indices)
    factor_index = np.random.randint(num_factors)

    # Sample two mini batches of latent variables.
    factors = _sample_factors(
        evaluator.latent_factor_indices, evaluator.factor_sizes, evaluator.batch_size
    )

    # Fix the selected factor across mini-batch.
    factors[:, factor_index] = factors[0, factor_index]

    # Obtain the observations.
    observations = _sample_observations_from_factors(
        factors, evaluator.latent_factor_indices, evaluator.factor_sizes, evaluator.imgs
    )
    observations = (
        torch.from_numpy(observations).float().view(evaluator.batch_size, 1, 64, 64)
    )

    with torch.no_grad():
        representations, _ = evaluator.model.encoder(observations)
    local_variances = np.var(representations.numpy(), axis=0, ddof=1)
    arg_min = np.argmin(local_variances[active_dims] / global_variances[active_dims])
    return factor_index, arg_min


def _compute_variances(evaluator, batch_size):
    factors = _sample_factors(
        evaluator.latent_factor_indices, evaluator.factor_sizes, batch_size
    )
    observations = _sample_observations_from_factors(
        factors, evaluator.latent_factor_indices, evaluator.factor_sizes, evaluator.imgs
    )

    observations = torch.from_numpy(observations).float().view(batch_size, 1, 64, 64)
    with torch.no_grad():
        representations, _ = evaluator.model.encoder(observations)

    # representations = np.transpose(representations)
    assert representations.shape[0] == batch_size
    return np.var(representations.numpy(), axis=0, ddof=1)


def _prune_dims(variances, threshold=0.0):
    scale_z = np.sqrt(variances)
    return scale_z >= threshold


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

    factors1 = _sample_factors(
        evaluator.latent_factor_indices, evaluator.factor_sizes, evaluator.batch_size
    )
    factors2 = _sample_factors(
        evaluator.latent_factor_indices, evaluator.factor_sizes, evaluator.batch_size
    )

    # Ensure sampled coordinate is the same across pairs of samples.
    factors2[:, index] = factors1[:, index]

    # Transform latent variables to observation space.
    observation1 = _sample_observations_from_factors(
        factors1,
        evaluator.latent_factor_indices,
        evaluator.factor_sizes,
        evaluator.imgs,
    )
    observation1 = torch.from_numpy(observation1).float().view(64, 1, 64, 64)
    observation2 = _sample_observations_from_factors(
        factors2,
        evaluator.latent_factor_indices,
        evaluator.factor_sizes,
        evaluator.imgs,
    )
    observation2 = torch.from_numpy(observation2).float().view(64, 1, 64, 64)

    # Compute representations based on the observations.
    with torch.no_grad():
        representation1, _ = evaluator.model.encoder(observation1)
        representation2, _ = evaluator.model.encoder(observation2)

    # Compute the feature vector based on differences in representation.
    feature_vector = np.mean(
        np.abs(representation1.numpy() - representation2.numpy()), axis=0
    )
    return index, feature_vector


def ls_sample_factors(latent_factor_indices, factor_sizes, num):
    factors = np.zeros(shape=(num, len(latent_factor_indices)), dtype=np.int64)
    for pos, i in enumerate(latent_factor_indices):
        factors[:, pos] = np.random.randint(factor_sizes[i], size=num)
    return factors


def _sample_observations_from_factors(
        factors, latent_factor_indices, factor_sizes, images
):
    num_samples = factors.shape[0]
    num_factors = len(latent_factor_indices)

    all_factors = np.zeros(shape=(num_samples, num_factors), dtype=np.int64)
    all_factors[:, latent_factor_indices] = factors

    # Complete all the other factors
    observation_factor_indices = [
        i for i in range(num_factors) if i not in latent_factor_indices
    ]

    for i in observation_factor_indices:
        all_factors[:, i] = np.random.randint(factor_sizes[i], size=num_samples)

    factor_bases = np.prod(factor_sizes) / np.cumprod(factor_sizes)
    indices = np.array(np.dot(all_factors, factor_bases), dtype=np.int64)

    return np.expand_dims(images[indices].astype(np.float32), axis=3)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate")
    parser.add_argument(
        "--model", type=str, default="factor_vae", help="which model to evaluate"
    )
    parser.add_argument(
        "--metric", type=str, default="downstream", help="which metric to evaluate"
    )
    parser.add_argument(
        "--parameter", type=int, default=1, help="the parameter is the regularizer"
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="which seed the model is trained on"
    )
    parser.add_argument(
        "--prediction_model",
        type=str,
        default="logistic",
        help="which model to use in downstream",
    )

    args = parser.parse_args()

    # run this file from the script folder
    data_path = "../data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

    n_train = 10000
    n_test = 5000

    path = "../results/dsprites/"

    metrics = {"train": None, "test": None}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = path + args.model + f"/parameter_{args.parameter}/seed_{args.seed}"
    model_data = torch.load(model_path + "/model.pt", map_location=torch.device("cpu"))
    model = ConvVAE(model_data["opt"]).to(device)
    model.load_state_dict(state_dict=model_data["model"])
    opt = model_data["opt"]

    evaluate = Evaluator(data_path=data_path, model=model, opt=opt, batch_size=64)

    if args.metric == "beta":
        metric_train, metric_test = evaluate.get_beta_metric(n_train, n_test)
        metrics["train"] = metric_train
        metrics["test"] = metric_test

    elif args.metric == "factor":
        metric_train, metric_test = evaluate.get_factor_metric(
            n_train, n_test, num_variance_estimate=10000
        )
        metrics["train"] = metric_train
        metrics["test"] = metric_test

    elif args.metric == "mig":
        metrics["mig"] = evaluate.get_mig_metric(num_bins=10)

    elif args.metric == "downstream":
        args.metric = args.metric + "_" + args.prediction_model
        metrics = evaluate.get_downstream(num_train=[10])

    else:
        raise NotImplementedError("Metric not implemented")

    metric_path = (
            path
            + "metrics/"
            + args.metric
            + "_"
            + args.model
            + f"_{args.parameter}_{args.seed}.json"
    )
    with open(metric_path, "w") as f:
        json.dump(metrics, f)
