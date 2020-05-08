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

    Seetharaman, Prem. Bootstrapping the Learning Process for Computer Audition. 
    Diss. Northwestern University, 2019.
    
    Args:
        audio_signal (AudioSignal): AudioSignal object which will be used to compute
          the mask over which to compute the confidence measure. This can be None, if
          and only if ``representation`` is passed as a keyword argument to this 
          function.
        features (np.ndarray): Numpy array containing the features to be clustered. 
          Should have the same dimensions as the representation.
        n_sources (int): Number of sources to cluster the features into.
        threshold (int, optional): Threshold by loudness. Points below the threshold are
          excluded from being used in the confidence measure. Defaults to 95.
        kwargs: Keyword arguments to `_get_loud_bins_mask`. Namely, representation can
          go here as a keyword argument.

    Returns:
        float: Confidence given by Jensen-Shannon divergence.
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
    Calculates the clusterability of an embedding space by looking at the
    strength of the assignments of each point to a specific cluster. The 
    more points that are "in between" clusters (e.g. no strong assignmment),
    the lower the clusterability.

    References:

    Seetharaman, Prem, Gordon Wichern, Jonathan Le Roux, and Bryan Pardo. 
    “Bootstrapping Single-Channel Source Separation via Unsupervised Spatial 
    Clustering on Stereo Mixtures”. 44th International Conference on Acoustics, 
    Speech, and Signal Processing, Brighton, UK, May, 2019

    Seetharaman, Prem. Bootstrapping the Learning Process for Computer Audition. 
    Diss. Northwestern University, 2019.
    
    Args:
        audio_signal (AudioSignal): AudioSignal object which will be used to compute
          the mask over which to compute the confidence measure. This can be None, if
          and only if ``representation`` is passed as a keyword argument to this 
          function.
        features (np.ndarray): Numpy array containing the features to be clustered. 
          Should have the same dimensions as the representation.
        n_sources (int): Number of sources to cluster the features into.
        threshold (int, optional): Threshold by loudness. Points below the threshold are
          excluded from being used in the confidence measure. Defaults to 95.
        kwargs: Keyword arguments to `_get_loud_bins_mask`. Namely, representation can
          go here as a keyword argument.
    
    Returns:
        float: Confidence given by posteriors.
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
    """
    Uses the silhouette score to compute the clusterability of the feature space.

    The Silhouette Coefficient is calculated using the 
    mean intra-cluster distance (a) and the mean nearest-cluster distance (b) 
    for each sample. The Silhouette Coefficient for a sample is (b - a) / max(a, b). 
    To clarify, b is the distance between a sample and the nearest cluster 
    that the sample is not a part of. Note that Silhouette Coefficient is 
    only defined if number of labels is 2 <= n_labels <= n_samples - 1.

    References:

    Seetharaman, Prem. Bootstrapping the Learning Process for Computer Audition. 
    Diss. Northwestern University, 2019.

    Peter J. Rousseeuw (1987). “Silhouettes: a Graphical Aid to the 
    Interpretation and Validation of Cluster Analysis”. Computational and 
    Applied Mathematics 20: 53-65.
    
    Args:
        audio_signal (AudioSignal): AudioSignal object which will be used to compute
          the mask over which to compute the confidence measure. This can be None, if
          and only if ``representation`` is passed as a keyword argument to this 
          function.
        features (np.ndarray): Numpy array containing the features to be clustered. 
          Should have the same dimensions as the representation.
        n_sources (int): Number of sources to cluster the features into.
        threshold (int, optional): Threshold by loudness. Points below the threshold are
          excluded from being used in the confidence measure. Defaults to 95.
        kwargs: Keyword arguments to `_get_loud_bins_mask`. Namely, representation can
          go here as a keyword argument.
        max_points (int, optional): Maximum number of points to compute the Silhouette
          score for. Silhouette score is a costly operation. Defaults to 1000.
    
    Returns:
        float: Confidence given by Silhouette score.
    """
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
    """
    Computes the clusterability of the feature space by comparing the absolute
    size of each cluster.
    
    References:

    Seetharaman, Prem, Gordon Wichern, Jonathan Le Roux, and Bryan Pardo. 
    “Bootstrapping Single-Channel Source Separation via Unsupervised Spatial 
    Clustering on Stereo Mixtures”. 44th International Conference on Acoustics, 
    Speech, and Signal Processing, Brighton, UK, May, 2019

    Seetharaman, Prem. Bootstrapping the Learning Process for Computer Audition. 
    Diss. Northwestern University, 2019.
    
    Args:
        audio_signal (AudioSignal): AudioSignal object which will be used to compute
          the mask over which to compute the confidence measure. This can be None, if
          and only if ``representation`` is passed as a keyword argument to this 
          function.
        features (np.ndarray): Numpy array containing the features to be clustered. 
          Should have the same dimensions as the representation.
        n_sources (int): Number of sources to cluster the features into.
        threshold (int, optional): Threshold by loudness. Points below the threshold are
          excluded from being used in the confidence measure. Defaults to 95.
        kwargs: Keyword arguments to `_get_loud_bins_mask`. Namely, representation can
          go here as a keyword argument.
    
    Returns:
        float: Confidence given by size of smallest cluster.
    """
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
    """
    Computes the clusterability in two steps:

    1. Cluster the feature space using KMeans into assignments
    2. Compute the Whitened K-Means loss between the features and the assignments.
    
    Args:
        audio_signal (AudioSignal): AudioSignal object which will be used to compute
          the mask over which to compute the confidence measure. This can be None, if
          and only if ``representation`` is passed as a keyword argument to this 
          function.
        features (np.ndarray): Numpy array containing the features to be clustered. 
          Should have the same dimensions as the representation.
        n_sources (int): Number of sources to cluster the features into.
        threshold (int, optional): Threshold by loudness. Points below the threshold are
          excluded from being used in the confidence measure. Defaults to 95.
        kwargs: Keyword arguments to `_get_loud_bins_mask`. Namely, representation can
          go here as a keyword argument.
    
    Returns:
        float: Confidence given by whitened k-means loss.
    """
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
    """
    Computes the clusterability in two steps:

    1. Cluster the feature space using KMeans into assignments
    2. Compute the classic deep clustering loss between the features and the assignments.
    
    Args:
        audio_signal (AudioSignal): AudioSignal object which will be used to compute
          the mask over which to compute the confidence measure. This can be None, if
          and only if ``representation`` is passed as a keyword argument to this 
          function.
        features (np.ndarray): Numpy array containing the features to be clustered. 
          Should have the same dimensions as the representation.
        n_sources (int): Number of sources to cluster the features into.
        threshold (int, optional): Threshold by loudness. Points below the threshold are
          excluded from being used in the confidence measure. Defaults to 95.
        kwargs: Keyword arguments to `_get_loud_bins_mask`. Namely, representation can
          go here as a keyword argument.
    
    Returns:
        float: Confidence given by deep clustering loss.
    """
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
