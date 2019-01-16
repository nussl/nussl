import torch
import torch.nn as nn
import numpy as np
from itertools import permutations

class DeepClusteringLoss(nn.Module):
    def __init__(self):
        """
        Computes the deep clustering loss with weights. Equation (7) in [1].

        References:
            [1] Wang, Z. Q., Le Roux, J., & Hershey, J. R. (2018, April).
            Alternative Objective Functions for Deep Clustering.
            In Proc. IEEE International Conference on Acoustics,  Speech
            and Signal Processing (ICASSP).
        """
        super(DeepClusteringLoss, self).__init__()

    def forward(self, embedding, assignments, weights):
        batch_size = embedding.shape[0]
        embedding_size = embedding.shape[-1]
        num_sources = assignments.shape[-1]

        weights = weights.view(batch_size, -1, 1)
        embedding = embedding.view(batch_size, -1, embedding_size)
        assignments = assignments.view(batch_size, -1, num_sources)

        assignments = weights.expand_as(assignments) * assignments
        embedding = weights.expand_as(embedding) * embedding
        norm = ((((weights) ** 2)).sum(dim=1) ** 2).sum()

        #norm = (((embedding_size ** 2) * (num_weights ** 2)) -
        #       2 * (embedding_size * (num_weights ** 2)) + (
        #            num_weights ** 2))

        vTv = ((embedding.transpose(2, 1) @ embedding) ** 2).sum()
        vTy = ((embedding.transpose(2, 1) @ assignments) ** 2).sum()
        yTy = ((assignments.transpose(2, 1) @ assignments) ** 2).sum()
        loss = (vTv - 2 * vTy + yTy) / norm.detach()
        return loss

class PermutationInvariantLoss(nn.Module):
    def __init__(self, loss_function):
        super(PermutationInvariantLoss, self).__init__()
        self.loss_function = loss_function
        self.loss_function.reduce = False
        
    def forward(self, estimates, targets):
        num_batch, sequence_length, num_frequencies, num_sources = estimates.shape
        estimates = estimates.view(num_batch, sequence_length*num_frequencies, num_sources)
        targets = targets.view(num_batch, sequence_length*num_frequencies, num_sources)
        
        losses = []
        for p in permutations(range(num_sources)):
            loss = self.loss_function(estimates[:, :, list(p)], targets)
            loss = loss.mean(dim=1).mean(dim=-1)
            losses.append(loss)
        
        losses = torch.stack(losses,dim=-1)
        losses = torch.min(losses, dim=-1)[0]
        loss = torch.mean(losses)
        return loss

class WeightedL1Loss(nn.Module):
    def __init__(self, loss_function):
        super(WeightedL1Loss, self).__init__()
        self.loss_function = loss_function
        self.loss_function.reduce = False
        
    def forward(self, estimates, targets):
        shape = targets.shape
        weights = shape[-1]*nn.functional.normalize(1./torch.sum(estimates.view(shape[0], -1, shape[-1]) > 0, dim=1).float(), dim=-1, p=1).unsqueeze(1)
        loss = torch.mean(self.loss_function(estimates, targets).view(shape[0], -1, shape[-1]) * weights)
        return loss

