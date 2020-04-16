import torch
import torch.nn as nn
import numpy as np
import gpytorch

class GaussianMixtureTorch(nn.Module):
    def __init__(self, n_components, n_iter=5, covariance_type='diag',
                 covariance_init=1.0, reg_covar=1e-4):
        """
        Initializes a Gaussian mixture model with n_clusters. 
        
        Args:
            n_components (int): Number of components.
            n_iter (int, optional): Number of EM iterations. Defaults to 5.
            covariance_type (str, optional): Covariance type. 
            String describing the type of covariance parameters to use.

            Must be one of:
            'full'
                each component has its own general covariance matrix (this case
                is harder to fit in EM than the others and isn't recommended at
                the moment
            'tied'
                all components share the same general covariance matrix
            'diag'
                each component has its own diagonal covariance matrix
            'spherical'
                each component has its own single variance

            Defaults to 'diag'.

            covariance_init (float, optional): Initial covariance for all
            features and all clusters. Defaults to 0.1.

            reg_covar (float, optional): Regularization amount to add to
            covariance matrix.
        """
        self.n_components = n_components
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.covariance_init = covariance_init
        self.reg_covar = reg_covar

        super().__init__()

    def _m_step(self, X, resp):
        """
        Takes a maximization step on the data X.
        
        Args:
            X (torch.Tensor): Data, shape (n_batch, n_samples, n_features)
            resp (torch.Tensor): Responsibilities each Gaussian has for 
            each sample. (n_batch, n_samples, n_components)
        """
        n_batch, n_samples, n_features = X.shape
        _, _, n_components = resp.shape
        resp = resp.view(n_batch, n_samples, n_components, 1)
        X = X.view(n_batch, n_samples, 1, n_features)

        # update means
        _top = (resp * X).sum(dim=1, keepdims=True)
        _bottom = resp.sum(dim=1, keepdims=True)
        means = _top / _bottom

        # update covariance
        diff = X - means
        diff = diff * resp
        covariance = diff.permute(0, 2, 3, 1) @ diff.permute(0, 2, 1, 3)
        covariance = covariance.unsqueeze(1) / _bottom[..., None]
        covariance = covariance.squeeze(1).clamp(min=self.reg_covar)
        covariance = self._enforce_covariance_type(covariance)

        # update prior
        prior = _bottom

        return means.squeeze(1), covariance, prior

    @staticmethod
    def _e_step(X, means, covariance):
        """
        Takes the expectation of X. Returns the log probability of X under each
        Gaussian in the mixture model.
        
        Args:
            X (torch.Tensor): Data, shape (n_batch, n_samples, n_features)
            means (torch.Tensor): Means, shape (n_batch, n_components, n_features)
            covariance (torch.Tensor): (n_batch, n_components, n_features, n_features)
        """
        n_batch, n_samples, n_features = X.shape
        _, n_components, _ = means.shape

        X = X.view(n_batch, n_samples, 1, n_features)
        means = means.view(n_batch, 1, n_components, n_features)
        covariance = covariance.view(
            n_batch, 1, n_components, n_features, n_features)

        mvn = gpytorch.distributions.MultivariateNormal(
            means, covariance_matrix=covariance
        )
        log_prob = mvn.log_prob(X)
        prob = torch.exp(log_prob) + 1e-8
        resp = nn.functional.normalize(prob, p=1, dim=-1)
        return resp, log_prob

    def _enforce_covariance_type(self, covariance):
        n_features = covariance.shape[-1]
        diag_mask = torch.eye(n_features, device=covariance.device)
        diag_mask = diag_mask.reshape(1, 1, n_features, n_features)
        covariance = covariance * diag_mask

        if 'spherical' in self.covariance_type:
            covariance[..., :, :] = (
                covariance.mean(dim=[-2, -1], keepdims=True)
            )
            covariance = covariance * diag_mask

        if 'tied' in self.covariance_type:
            covariance[..., :, :] = covariance.mean(
                dim=1, keepdims=True
            )
        return covariance

    def init_params(self, X, means=None, covariance=None):
        """
        Initializes Gaussian parameters.
        
        Args:
            X (torch.Tensor): Data, shape (n_batch, n_samples, n_features)
            means (torch.Tensor): Means, shape (n_batch, n_components, n_features). Defaults
            to None.
            covariance (torch.Tensor): (n_batch, n_components, n_features, n_features) 
                or (n_batch, n_components, n_features).
            Defaults to None.
        """
        if means is None:
            sampled = X.new(
                X.shape[0], self.n_components).random_(0, X.shape[1])
            sampled += X.new(np.arange(0, X.shape[0])).unsqueeze(
                1).expand(-1, sampled.shape[1]) * X.shape[1]
            sampled = sampled.long()
            means = torch.index_select(
                X.view(-1, X.shape[-1]), 0, sampled.view(-1)).view(
                X.shape[0], sampled.shape[-1], -1)

        if covariance is None:
            covariance = X.new(
                X.shape[0], self.n_components, X.shape[-1]).fill_(
                self.covariance_init).clone()

        if len(covariance.shape) < 4:
            covariance = covariance.unsqueeze(-1).expand(-1, -1, -1, X.shape[-1])

        covariance = self._enforce_covariance_type(covariance.clone())

        return means, covariance

    def forward(self, data, means=None, covariance=None):
        """
        Does a forward pass of the GMM.
        
        Args:
            data (torch.Tensor): Data, shape is (n_batch, ..., n_features)
            means (torch.Tensor): Means, shape (n_batch, n_components, n_features). 
              Defaults to None.
            covariance (torch.Tensor): (n_batch, n_components, n_features, n_features) 
              or (n_batch, n_components, n_features). 
        
        Returns:
            dict: Containing keys 'resp', 'log_prob', 'means', 'covariance', 'prior'.
        """
        shape = data.shape
        data = data.view(shape[0], -1, shape[-1])
        means, covariance = self.init_params(data, means, covariance)

        resp = log_prob = prior = None
        for i in range(self.n_iter):
            resp, log_prob = self._e_step(data, means, covariance)
            means, covariance, prior = self._m_step(data, resp)

        return {
            'resp': resp.view(shape[:-1] + (-1,)),
            'log_prob': log_prob.view(shape[:-1] + (-1,)),
            'means': means,
            'covariance': covariance,
            'prior': prior
        }
