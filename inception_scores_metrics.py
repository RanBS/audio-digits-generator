"""
Calculating Frechet Inception Score metrics for a generative model.
"""

import numpy as np
import torch

from classifier import model as classifier_model
from classifier import dataset


def calc_features_mu_sigma(classifier, dataset):
    """
        Calculates the mean vector and sigma matrix of the features of a dataset.
        The features are received from a intermediate layer of the classifier.
        ====
        Arguments:
            classifer - a trained classifier for spectograms amplitudes.
            dataset   - a dataset on which we calculate the mean and sigma.
        Output:
            mu.
            sigma.
    """
    features = []
    for d in dataset:
        d = torch.unsqueeze(d['image'], 0)
        features.append(classifier.get_features(d).view(1, -1).cpu().data.numpy().squeeze())

    mu = np.mean(features, 0)

    var_vec = np.var(features, 0)
    sigma = np.diag(var_vec)

    return mu, sigma


def calculate_stats(classifier, real_dataset, fake_dataset):
    """
        Evaluates the performance of a generative model.
        ====
        Arguments:
            classifer - a trained classifier for spectograms amplitudes.
            real_dataset - a real spectograms dataset.
            fake_dataset - a generated spectograms dataset.
        Output:
            Frechet Inception Score.
            The norm of sigma, which is the variance matrix of the features (of fake samples).
    """
    mu_real, sigma_real = calc_features_mu_sigma(classifier, real_dataset)
    mu_fake, sigma_fake = calc_features_mu_sigma(classifier, fake_dataset)
    return ((np.linalg.norm(mu_real - mu_fake, 2) ** 2) + np.trace(
        sigma_real + sigma_fake - 2 * (sigma_real @ sigma_fake) ** 0.5)) / np.size(mu_fake), np.linalg.norm(sigma_fake,
                                                                                                            2)


if __name__ == '__main__':
    classifier = classifier_model.ConvNetwork()
    classifier.load_state_dict(torch.load('./classifier/weights.pth'))
    classifier.eval()

    for exp in ['1', '2', '3', '4', 'vae']:
        in_scores = []
        var = []
        for label in range(10):
            real_dataset = dataset.DigitsAudioDataset(
                './dataset/data_for_metrics/real_spectograms_amplitude/' + str(label))
            exp_dataset = dataset.DigitsAudioDataset('./dataset/data_for_metrics/exp ' + exp + '/' + str(label))
            in_, var_ = calculate_stats(classifier, real_dataset, exp_dataset)
            in_scores.append(in_)
            var.append(var_)

        print("exp " + exp + " - Frechet Inception Score:", np.mean(in_scores))
        print("exp " + exp + " - Diversity Score:", np.mean(var))
