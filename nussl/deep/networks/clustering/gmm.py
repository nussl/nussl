import torch
import torch.nn as nn
import numpy as np

class ScaleLayer(nn.Module):
    def __init__(self, init_value=1.0, init_bias=0.0, bias=False):
        super().__init__()
        self.scale = nn.Parameter(
            torch.FloatTensor([init_value]), requires_grad=True
        )
        if bias:
            self.bias = nn.Parameter(
                torch.FloatTensor([init_bias]), requires_grad=True
            )
        else:
            self.bias = init_bias    
        
    def forward(self, data):
        return (data * self.scale) + self.bias
    
class GMM(nn.Module):
    def __init__(self, n_clusters, n_iterations=5, covariance_type='diag', covariance_init=1.0):
        """
        Args:
            n_clusters:
            n_iterations:
            covariance_type:
            covariance_init:
        """
        super(GMM, self).__init__()
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.covariance_init = covariance_init
        self.covariance_type = covariance_type
        self.scale = ScaleLayer(init_value=1.0, bias=True)
       
    def init_means(self, data):
        """
        Args:
            data:

        Returns:

        """
        sampled = data.new(data.shape[0], self.n_clusters).random_(0, data.shape[1])
        sampled = data.new(np.arange(0, data.shape[0])).unsqueeze(1).expand(-1, sampled.shape[1])*data.shape[1] + sampled
        sampled = sampled.long()
        means = torch.index_select(data.view(-1, data.shape[-1]), 0, sampled.view(-1)).view(data.shape[0], sampled.shape[-1], -1)
        return means

    def init_var_and_prior(self, data):
        var = data.new(data.shape[0], self.n_clusters, data.shape[-1]).fill_(self.covariance_init)
        pi = data.new(data.shape[0], self.n_clusters).fill_(1./self.n_clusters)
        return var, pi
        
    def update_parameters(self, posteriors, data, weights):
        """
        Args:
            posteriors:
            data:
            weights:

        Returns:

        """
        if not isinstance(weights, float):
            weights = weights.unsqueeze(1).unsqueeze(-1)
            num_examples = weights.sum(dim=2)
        else:
            num_examples = data.shape[1]

        posteriors = posteriors.unsqueeze(-1).expand(-1, -1, -1, 1)
        posteriors = posteriors * weights #data weighting
        
        cluster_sizes = torch.sum(posteriors, dim=2)
        updated_pi = (cluster_sizes / num_examples).squeeze(-1)
        updated_pi = nn.functional.normalize(updated_pi, p=1, dim=-1)
        
        data = data.unsqueeze(1).expand(-1, 1, -1, -1)
        weighted_embeddings = posteriors * data
        updated_means = torch.sum(weighted_embeddings, dim=2) / (cluster_sizes + 1e-7)
        
        distance = data - updated_means.unsqueeze(2).expand(-1, -1, 1, -1)
        distance = posteriors * torch.pow(distance, 2)
        updated_var = torch.sum(distance, dim=2)
        updated_var = self._enforce_covariance_type(updated_var)
            
        return updated_means, updated_var, updated_pi

    def _enforce_covariance_type(self, var):
        if 'tied' in self.covariance_type:
            var = torch.mean(var, dim=1, keepdim=True).expand(-1, var.shape[1], -1)

        if 'spherical' in self.covariance_type:
            var = torch.mean(var, dim=-1, keepdim=True).expand(-1, -1, var.shape[-1])

        if 'fix' in self.covariance_type:
            var = var.clone()
            var[:, :, :] = self.covariance_init

        return var

    
    def update_posteriors(self, likelihoods):
        """

        Args:
            likelihoods:

        Returns:

        """
        #log-sum-exp trick https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
        max_value = likelihoods.max(dim=1, keepdim=True)[0]
        likelihoods_sum = max_value + torch.log((likelihoods - max_value).exp().sum(dim=1, keepdim=True))
        posteriors = (likelihoods - likelihoods_sum).exp()
        return posteriors, likelihoods
    
    def update_likelihoods(self, data, means, var, pi):
        num_batch = data.shape[0]
        num_examples = data.shape[1] * data.shape[2] * data.shape[3]
        num_features = data.shape[-1]
        num_clusters = means.shape[1]

        inv_covariance =  1 / (var + 1e-6)
        data = data.view(num_batch, 1, num_examples, num_features)
        means = means.unsqueeze(2).expand(-1, -1, 1, -1)
        distance = torch.pow(data - means, 2)
        distance = distance.view(-1, num_examples, num_features)
        inv_covariance = inv_covariance.view(-1, num_clusters, num_features, 1)

        if inv_covariance.shape[0] == 1:
            inv_covariance = inv_covariance.expand(num_batch, -1, -1, -1)
            inv_covariance = inv_covariance.reshape(-1, num_features, 1)
        
        distance = -.5 * torch.bmm(distance, inv_covariance)
        distance = distance.view(num_batch, num_clusters, num_examples)
        
        #log of determinant of 2*pi*variance, which is diagonal -> sum of log of 2 * pi * variance
        coeff = -.5 * torch.log(2 * np.pi * var).sum(dim=-1)
        prior = torch.log(pi)
        likelihoods = prior.unsqueeze(-1) + coeff.unsqueeze(-1) + distance
        return likelihoods
    
    def forward(self, data, means=None, var=None, pi=None, weights=1.0):
        """

        Args:
            data:
            parameters:
            weights:

        Returns:

        """
        shape = list(data.shape)

        if means is None:
            means = self.init_means(data)
        if var is None:
            var, pi = self.init_var_and_prior(data)
        else:
            _,  pi = self.init_var_and_prior(data)
            var = self._enforce_covariance_type(var)
        

        for i in range(self.n_iterations):
            likelihoods = self.update_likelihoods(data, means, var, pi)
            posteriors, likelihoods = self.update_posteriors(likelihoods)
            means, var, pi = self.update_parameters(posteriors, data, weights)
            var = var + 1e-6

        likelihoods = self.update_likelihoods(data, means, var, pi)
        posteriors, likelihoods = self.update_posteriors(likelihoods)
        shape[-1] = posteriors.shape[1]
        posteriors = posteriors.permute(0, 2, 1).view(shape)
        likelihoods = likelihoods.permute(0, 2, 1).view(shape)
        likelihoods = torch.sigmoid(self.scale(likelihoods))
        return likelihoods