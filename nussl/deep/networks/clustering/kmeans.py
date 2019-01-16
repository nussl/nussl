import torch
import torch.nn as nn
import numpy as np

class ScaleLayer(nn.Module):
    def __init__(self, init_value=1.0, init_bias=0.0, bias=False):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor([init_bias]))
        else:
            self.bias = init_bias    
        
    def forward(self, input):
        return (input * self.scale) + self.bias

class KMeans(nn.Module):
    def __init__(self, n_clusters, alpha, n_iterations=5, init_method='random'):
        super(KMeans, self).__init__()
        self.add_module('scale', ScaleLayer(init_value=alpha))
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.init_method = init_method
        allowed_methods = ['random', 'kmeans++']
        if self.init_method not in allowed_methods:
            raise ValueError('init_method \'%s\' not supported. Choose from [%s]' % (init_method, ', '.join(allowed_methods)))
            
    def initialize_means(self, data):
        means = []
        if self.init_method == 'random':
            sampled = data.new(data.shape[0], self.n_clusters).random_(0, data.shape[1])
            sampled = data.new(np.arange(0, data.shape[0])).unsqueeze(1).expand(-1, sampled.shape[1])*data.shape[1] + sampled
            sampled = sampled.long()
            means = torch.index_select(data.view(-1, data.shape[-1]), 0, sampled.view(-1)).view(data.shape[0], sampled.shape[-1], -1)
        elif self.init_method == 'kmeans++':
            raise NotImplementedError('Not implemented yet')
        return means
        
    def update_means(self, assignments, data, weights):
        assignments = assignments.unsqueeze(2).expand(-1, -1, 1, -1)
        data = data.unsqueeze(-1).expand(-1, -1, -1, 1)
        if not isinstance(weights, float):
            weights = weights.unsqueeze(-1).unsqueeze(-1)
        weighted_embeddings = assignments * weights * data
        updated_means = (torch.sum(weighted_embeddings, dim=1) / torch.sum(assignments * weights, dim=1))
        return updated_means
    
    def update_assignments(self, data, means):
        data = data.unsqueeze(-1).expand(-1, -1, -1, 1)
        means = means.unsqueeze(1).expand(-1, 1, -1, -1)
        distance = -torch.pow(data - means, 2).sum(dim=2)
        assignments = nn.functional.softmax(self.scale(distance), dim=-1)
        return assignments
    
    def forward(self, data, parameters=None):
        if parameters is None:
            means = None
        if parameters['means'] is None:
            means = self.initialize_means(data)
        if means.shape[0] != data.shape[0]:
            means = means.expand(data.shape[0], -1, -1)
        means = means.permute(0, 2, 1)

        for i in range(self.n_iterations):
            assignments = self.update_assignments(data, means)
            means = self.update_means(assignments, data, weights)
        
        assignments = self.update_assignments(data, means)
        means = means.permute(0, 2, 1)
        return {'assignments': assignments,
                'parameters': means}