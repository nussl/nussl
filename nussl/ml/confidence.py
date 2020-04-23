"""
There are ways to measure the quality of a separated source without
requiring ground truth. These functions operate on the output of
clustering-based separation algorithms and work by analyzing
the clusterability of the feature space used to generate the
separated sources.
"""

from sklearn.metrics import silhouette_samples
import numpy as np
from .cluster import KMeans, GaussianMixture
from scipy.special import logsumexp
from .train import loss
import torch

def softmax(x, axis=None):
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))

def jensen_shannon_divergence(gmm_p, gmm_q, n_samples=10**5):
    """
    Compute Jensen-Shannon (JS) divergence between two Gaussian Mixture Models via 
    sampling. JS divergence is also known as symmetric Kullback-Leibler divergence.
    JS divergence has no closed form in general for GMMs, thus we use sampling to 
    compute it.

    Args:
        gmm_p (GaussianMixture): A GaussianMixture class fit to some data.
        gmm_q (GaussianMixture): Another GaussianMixture class fit to some data.
        n_samples (int): Number of samples to use to estimate JS divergence. 

    Returns:
        JS divergence between gmm_p and gmm_q
    """
    X = gmm_p.sample(n_samples)[0]
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    log_mix_X = np.logaddexp(log_p_X, log_q_X)

    Y = gmm_q.sample(n_samples)[0]
    log_p_Y = gmm_p.score_samples(Y)
    log_q_Y = gmm_q.score_samples(Y)
    log_mix_Y = np.logaddexp(log_p_Y, log_q_Y)

    return (log_p_X.mean() - (log_mix_X.mean() - np.log(2))
            + log_q_Y.mean() - (log_mix_Y.mean() - np.log(2))) / 2

def _get_loud_bins_mask(threshold, audio_signal=None, representation=None):
    if representation is None:
        representation = np.abs(audio_signal.stft())
    threshold = np.percentile(representation, threshold)
    mask = representation > threshold
    return mask, representation

def jensen_shannon_confidence(audio_signal, features, num_sources, threshold=95, 
                              n_samples=10**5, **kwargs):
    """
    Calculates the clusterability of a space by comparing a K-cluster GMM
    with a 1-cluster GMM on the same features. This function fits two
    GMMs to all of the points that are above the specified threshold (defaults
    to 95: 95th percentile of all the data). This saves on computation time and
    also allows one to have the confidence measure only focus on the louder
    more perceptually important points.

    References:

    Seetharaman, Prem, Gordon Wichern, Jonathan Le Roux, and Bryan Pardo. 
    “Bootstrapping Single-Channel Source Separation via Unsupervised Spatial 
    Clustering on Stereo Mixtures”. 44th International Conference on Acoustics, 
    Speech, and Signal Processing, Brighton, UK, May, 2019
    
    Args:
        audio_signal ([type]): [description]
        features ([type]): [description]
        n_sources ([type]): [description]
        threshold (int, optional): [description]. Defaults to 95.
    """
    mask, _ = _get_loud_bins_mask(threshold, audio_signal, **kwargs)
    embedding_size = features.shape[-1]
    features = features[mask].reshape(-1, embedding_size)

    one_component_gmm = GaussianMixture(1)
    n_component_gmm = GaussianMixture(num_sources)

    one_component_gmm.fit(features)
    n_component_gmm.fit(features)

    confidence = jensen_shannon_divergence(
        one_component_gmm, n_component_gmm, n_samples=n_samples)

    return confidence

def posterior_confidence(audio_signal, features, num_sources, threshold=95, 
                         **kwargs):
    """
    Simple 
    
    Args:
        audio_signal ([type]): [description]
        features ([type]): [description]
        num_sources ([type]): [description]
        threshold (int, optional): [description]. Defaults to 95.
    
    Returns:
        [type]: [description]
    """
    mask, _ = _get_loud_bins_mask(threshold, audio_signal, **kwargs)
    embedding_size = features.shape[-1]
    features = features[mask].reshape(-1, embedding_size)

    kmeans = KMeans(num_sources)
    distances = kmeans.fit_transform(features)

    confidence = softmax(-distances, axis=-1)

    confidence = (
        (num_sources * np.max(confidence, axis=-1) - 1) / 
        (num_sources - 1)
    )

    return confidence.mean()

def silhouette_confidence(audio_signal, features, num_sources, threshold=95, 
                          max_points=1000, **kwargs):
    mask, _ = _get_loud_bins_mask(threshold, audio_signal, **kwargs)
    embedding_size = features.shape[-1]
    features = features[mask].reshape(-1, embedding_size)

    if features.shape[0] > max_points:
        idx = np.random.choice(
            np.arange(features.shape[0]), max_points,
            replace=False)
        features = features[idx]
    
    kmeans = KMeans(num_sources)

    labels = kmeans.fit_predict(features)
    confidence = silhouette_samples(features, labels)

    return confidence.mean()

def loudness_confidence(audio_signal, features, num_sources, threshold=95, 
                        **kwargs):
    mask, _ = _get_loud_bins_mask(threshold, audio_signal, **kwargs)
    embedding_size = features.shape[-1]
    features = features[mask].reshape(-1, embedding_size)

    kmeans = KMeans(num_sources)
    labels = kmeans.fit_predict(features)

    source_shares = np.array(
        [(labels == i).sum() for i in range(num_sources)]
    ).astype(float)
    source_shares *= (1 / source_shares.sum())
    confidence = source_shares.min()

    return confidence

def whitened_kmeans_confidence(audio_signal, features, num_sources, threshold=95, 
                               **kwargs):
    mask, representation = _get_loud_bins_mask(threshold, audio_signal, **kwargs)
    embedding_size = features.shape[-1]
    features = features[mask].reshape(-1, embedding_size)
    weights = representation[mask].reshape(-1)

    kmeans = KMeans(num_sources)
    distances = kmeans.fit_transform(features)
    assignments = (distances == distances.max(axis=-1, keepdims=True))

    loss_func = loss.WhitenedKMeansLoss()

    features = torch.from_numpy(features).unsqueeze(0).float()
    assignments = torch.from_numpy(assignments).unsqueeze(0).float()
    weights = torch.from_numpy(weights).unsqueeze(0).float()

    loss_val = loss_func(features, assignments, weights).item()
    upper_bound = embedding_size + num_sources
    confidence = 1 - (loss_val / upper_bound)
    return confidence

def dpcl_classic_confidence(audio_signal, features, num_sources, threshold=95, 
                            **kwargs):
    mask, representation = _get_loud_bins_mask(threshold, audio_signal, **kwargs)
    embedding_size = features.shape[-1]
    features = features[mask].reshape(-1, embedding_size)
    weights = representation[mask].reshape(-1)

    kmeans = KMeans(num_sources)
    distances = kmeans.fit_transform(features)
    assignments = (distances == distances.max(axis=-1, keepdims=True))

    loss_func = loss.DeepClusteringLoss()

    features = torch.from_numpy(features).unsqueeze(0).float()
    assignments = torch.from_numpy(assignments).unsqueeze(0).float()
    weights = torch.from_numpy(weights).unsqueeze(0).float()

    loss_val = loss_func(features, assignments, weights).item()
    confidence = 1 - loss_val
    return confidence
