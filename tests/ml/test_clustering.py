"""
This file is a palceholder for if new clustering algorithms come into 
nussl at some point and need to be tested.
"""

from sklearn import datasets
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.preprocessing import StandardScaler
import sklearn
import numpy as np
import pytest
from nussl.ml import cluster
import matplotlib.pyplot as plt
import nussl


@pytest.fixture(scope="module")
def cluster_data():
    np.random.seed(0)

    # ============
    # Generate datasets. We choose the size big enough to see the scalability
    # of the algorithms, but not too big to avoid too long running times
    # ============
    n_samples = 1500
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                          noise=.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)

    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(n_samples=n_samples,
                                 cluster_std=[1.0, 2.5, 0.5],
                                 random_state=random_state)

    default_base = {'quantile': .3,
                    'eps': .3,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': 3,
                    'min_samples': 20,
                    'xi': 0.05,
                    'min_cluster_size': 0.1}

    data = [
        ('noisy_circles', noisy_circles, {'damping': .77, 'preference': -240,
                                          'quantile': .2, 'n_clusters': 2,
                                          'min_samples': 20, 'xi': 0.25}),
        ('noisy_moons', noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
        ('varied', varied, {'eps': .18, 'n_neighbors': 2,
                            'min_samples': 5, 'xi': 0.035, 'min_cluster_size': .2}),
        ('aniso', aniso, {'eps': .15, 'n_neighbors': 2,
                          'min_samples': 20, 'xi': 0.1, 'min_cluster_size': .2}),
        ('blobs', blobs, {}),
    ]
    yield data, default_base
