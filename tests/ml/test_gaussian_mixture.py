from nussl.ml.unfold import GaussianMixtureTorch
import torch
import numpy as np
from torch import nn
from sklearn.metrics import adjusted_mutual_info_score
from sklearn import mixture, cluster


def test_ml_gaussian_mixture():
    loc = torch.randn(1, 1, 3, 2)
    cov = torch.eye(2).view(1, 1, 1, 2, 2)
    cov = cov.repeat(1, 1, 3, 1, 1)

    for i in range(loc.shape[2]):
        loc[:, :, i, :] += (i * 10)
        cov[:, :, i, :, :] *= .1

    n_components = 3
    covariance_types = ['spherical', 'diag', 'tied', 'full']

    for covariance_type in covariance_types:
        mv = torch.distributions.MultivariateNormal(loc, covariance_matrix=cov)
        X = mv.sample((10, 1000)).view(10, 3000, 1, -1)

        labels = mv.log_prob(X).cpu().numpy().reshape(10, -1, n_components)
        labels = np.argmax(labels, axis=-1)

        X = X.view(10, 3000, -1)

        gmm = GaussianMixtureTorch(
            n_components=n_components, covariance_type=covariance_type)
        _loc = loc.view(1, 3, 2).expand(10, -1, -1)
        _cov = cov.view(1, 3, 2, 2).expand(10, -1, -1, -1)
        means, covariances = gmm.init_params(X, _loc, _cov)

        # with known means and covariances, ami should be perfect
        resp, prob = gmm._e_step(X, means, covariances)
        predictions = resp.cpu().numpy().reshape(10, -1, n_components)
        predictions = np.argmax(predictions, axis=-1)

        for nb in range(predictions.shape[0]):
            ami = adjusted_mutual_info_score(labels[nb], predictions[nb])
            assert np.allclose(ami, 1.0)

        # with random init, we compare ami with sklearn impl.
        # covariance_type = 'full' has some issues, i think due to init.

        if covariance_type != 'full':
            means, covariances = gmm.init_params(X)

            for i in range(50):
                assert (means.shape == (X.shape[0], n_components, X.shape[-1]))
                assert (covariances.shape == (
                    X.shape[0], n_components, X.shape[-1], X.shape[-1]))

                resp, prob = gmm._e_step(X, means, covariances)
                assert torch.allclose(
                    resp.sum(dim=-1, keepdims=True), torch.ones_like(resp))

                means, covariances, prior = gmm._m_step(X, resp)

            resp, prob = gmm._e_step(X, means, covariances)
            predictions = resp.cpu().numpy().reshape(10, -1, n_components)
            predictions = np.argmax(predictions, axis=-1)
            comps = []

            for nb in range(predictions.shape[0]):
                nussl_ami = adjusted_mutual_info_score(labels[nb], predictions[nb])

                sklearn_gmm = mixture.GaussianMixture(
                    n_components=n_components, covariance_type=covariance_type
                )

                npX = X[nb].cpu().numpy().reshape(-1, 2)
                sklearn_gmm.fit(npX)
                sklearn_predictions = sklearn_gmm.predict(npX)

                sklearn_ami = adjusted_mutual_info_score(
                    labels[nb].reshape(-1), sklearn_predictions)
                comps.append(nussl_ami >= sklearn_ami)

            assert sum(comps) >= len(comps) * .7

        forward_pass = gmm(X)

        assert forward_pass['resp'].shape == (X.shape[:-1] + (n_components,))
        assert forward_pass['log_prob'].shape == (X.shape[:-1] + (n_components,))
