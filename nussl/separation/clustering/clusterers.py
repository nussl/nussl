from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin
import numpy as np
from scipy.special import softmax
from sklearn import preprocessing
import scipy
from sklearn.metrics.pairwise import rbf_kernel

def gmm_js(gmm_p, gmm_q, n_samples=10**5):
    """
    Compute Jensen-Shannon (JS) divergence between two Gaussian Mixture Models via 
    sampling. JS divergence is also known as symmetric Kullback-Leibler divergence.

    JS divergence has no closed form in general for GMMs, thus we use sampling to 
    compute it.

    Parameters:
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

class KMeansConfidence(KMeans):
    def __init__(self, n_components, alpha=1.0, **kwargs):
        kwargs['n_clusters'] = n_components
        self.alpha = alpha
        super().__init__(**kwargs)

    def confidence(self, features):
        """
        Confidence for KMeans is: 
            x = (distance to centroids raised to power alpha)
            y = (average distance between centroids)
            confidence = y * softmax(x)
        """
        distances = super().transform(features) ** self.alpha
        distances /= distances.max()
        return softmax(distances, axis=-1)

    def predict_and_get_confidence(self, features):
        """
        Just wraps GaussianMixture.predict_proba, but saves the assignments so that
        confidence can be computed.
        """
        confidence = self.confidence(features)
        assignments = (confidence == confidence.max(axis=-1, keepdims=True)).astype(float)
        confidence = np.max(confidence, axis=-1)
        return assignments, confidence

class GaussianMixtureConfidence(GaussianMixture):
    def __init__(self, n_samples=10**5, alpha=1.0, **kwargs):
        """
        This class extends GaussianMixture from sklearn to include a confidence measure
        as laid out in:

        Seetharaman, Prem, Gordon Wichern, Jonathan Le Roux, and Bryan Pardo. 
        “Bootstrapping Single-Channel Source Separation via Unsupervised Spatial 
        Clustering on Stereo Mixtures”. 44th International Conference on Acoustics, 
        Speech, and Signal Processing, Brighton, UK, May, 2019

        Parameters:
            n_samples (int): Number of samples to use when computing JS divergence.
        """
        super().__init__(**kwargs)
        self.assignments = None
        self.n_samples = n_samples
        self.alpha = alpha

    def confidence(self, features):
        if self.assignments is None:
            raise RuntimeError("self.predict_proba must be run before computing confidence!")
        
        c1 = self.js_confidence(features)
        c2 = self.posterior_confidence()
        c3 = self.likelihood_confidence()
        c = (c1 * c2 * c3) ** self.alpha
        return c
        
    def js_confidence(self, features):
        """
        The first confidence measure computes the JS divergence between a 1-component
        GMM and a multi-component GMM. 
        """
        js_divergence = []
        for k in range(1, self.n_components):
            k_component_gmm = GaussianMixture(
                n_components=k, 
                covariance_type=self.covariance_type,
            ).fit(features)

            jsd = gmm_js(self, k_component_gmm, self.n_samples)
            js_divergence.append(jsd)

        return min(js_divergence)

    def posterior_confidence(self):
        """
        A confidence measure that emphasizes points close to the mean
        of Gaussians.
        """
        factor = float(self.n_components)
        return (factor*np.abs(np.max(self.assignments, axis=-1) - (1 / factor)))

    def likelihood_confidence(self):
        likelihoods = np.exp(self.log_likelihoods)
        likelihoods /= (likelihoods.max() + 1e-6)
        return likelihoods


    def predict_and_get_confidence(self, features):
        """
        Just wraps GaussianMixture.predict_proba, but saves the assignments so that
        confidence can be computed.
        """
        self.assignments = super().predict_proba(features)
        self.log_likelihoods = super().score_samples(features)
        confidence = self.confidence(features)
        return self.assignments, confidence
        
def fast_spectral_clustering(X, rank, weights=None, n_samples=1000, choices=None):
    """
    Implements the spectral clustering algorithm described in:

    Li, M., Lian, X. C., Kwok, J. T., & Lu, B. L. (2011, June). 
    Time and space efficient spectral clustering via column sampling. 
    In CVPR 2011 (pp. 2297-2304). IEEE.

    Args:
        X (np.ndarray): (n_samples, n_features) data to spectral cluster
        rank (int): Desired rank of spectral clustering. Probably don't set too high (<15).
        threshold (np.ndarray): bool, optional numpy array of the same shape as X. Contains
            bools so you can exclude things from sampling.
        n_samples (int): Number of columns to sample of affinity matrix.
        choices (np.ndarray): Indices to sample from X. 
    
    Returns:
        U (np.ndarray): Estimated projection of X onto eigenvectors of X
        S3 (np.ndarray): Estimated eigenvalues of X
        choices (np.ndarray): Sampled indices

    """
    if choices is None:
        choices = np.random.choice(X.shape[0], n_samples, p=weights.flatten(), replace=False)
    Z = X[choices]
    
    # Algorithm 1
    # compute top left block of kernel matrix - steps 1 - 3
    gamma = .5 #1 / ((1 / n_samples ** 2) * De.sum())
    A_11 = rbf_kernel(Z, Z, gamma=gamma)
    np.fill_diagonal(A_11, 0)

    D_star = np.diag(1 / np.sqrt(A_11.sum(axis=0) + 1e-6))
    M_star = D_star @ A_11 @ D_star

    # partial eigen-decomposition of M* - step 4
    S1, V1 = scipy.sparse.linalg.eigsh(M_star, k=rank, which='LM')
    S1 = np.maximum(0, S1) + 1e-8

    # step 5
    B = D_star @ V1 @ np.diag(1 / S1)
    S1 = np.diag(S1)

    # step 6-9
    Q = np.empty((X.shape[0], rank))
    block_size = min(100000, X.shape[0])
    for i in np.arange(0, X.shape[0], block_size):
        a = rbf_kernel(X[i:i+block_size], Z, gamma=gamma)
        Q[i:i+block_size] = a @ B

    # step 10 and 11
    D_hat = ((Q @ S1) @ (Q.T @ np.ones(Q.shape[0]))) ** 2
    D_hat = 1 / np.sqrt(np.sqrt(D_hat) + 1e-4)
    U = np.multiply(D_hat[:,None], Q)

    # orthogonalize U
    # Algorithm 2
    # step 1
    P = U.T @ U
    
    # step 2 full eigendecomposition
    S2, V2 = scipy.linalg.eigh(P)
    S2 = np.maximum(0, S2) + 1e-8
    S2_ = np.diag(np.sqrt(S2))

    # step 3
    B2 = S2_ @ (V2.T @ S1 @ V2) @ S2_

    # step 4
    S3, V3 = scipy.linalg.eigh(B2)
    S3 = np.maximum(0, S3) + 1e-8

    #after dropping the leading eigenvector of M, we perform
    #k-means clustering using the remaining c leading eigenvectors, 
    #and with each data row normalized to unit norm
    # step 5
    U = U @ V2 @ np.diag(1 / np.sqrt(S2)) @ V3
    U = U[:, ::-1]
    U = preprocessing.normalize(U, norm='l2')
    S3 = 1-S3[::-1]
    return U, S3, choices

class SpectralClusteringConfidence():
    def __init__(
        self,
        n_components,
        clustering_type='kmeans',
        n_samples=1000,
        max_rank=10,
        alpha=1.0,
        weights=None
    ):
        self.n_components = n_components
        self.n_samples = n_samples
        self.weights = weights / weights.sum()
        self.max_rank = max_rank
        self.alpha = alpha
        
        allowed_clustering_types = ['kmeans', 'gmm']
        if clustering_type not in allowed_clustering_types:
            raise ValueError(
                f"clustering_type = {clustering_type} not allowed!" 
                f"Use one of {allowed_clustering_types}."
            )
        
        if clustering_type == 'kmeans':
            self.clusterer = KMeansConfidence(
                n_components=n_components
            )
        elif clustering_type == 'gmm':
            self.clusterer = GaussianMixtureConfidence(
                n_components=n_components,
            )

    def fit(self, features):
        self.projection, self.eigenvalues, self.choices = fast_spectral_clustering(
            features, self.max_rank, self.weights, self.n_samples
        )
        self.clusterer.fit(self.projection)

    def confidence(self):
        lambdas = self.eigenvalues
        lambda_diff = np.abs(np.diff(lambdas))
        sort_lambda = np.sort(lambda_diff)[::-1]
        sort_index = np.argsort(lambda_diff)[::-1]
        i = list(sort_index).index(self.n_components)
        sort_lambda_threshold = np.minimum(sort_lambda, 15)
        confidence = sort_lambda[i] / sort_lambda_threshold.sum()
        return confidence ** self.alpha
        
    def predict_and_get_confidence(self, features):
        assignments, confidence = self.clusterer.predict_and_get_confidence(
            self.projection
        )
        confidence = self.confidence() * np.ones(confidence.shape)
        return assignments, confidence