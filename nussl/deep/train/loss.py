import torch
import torch.nn as nn
import numpy as np
from itertools import permutations, combinations

class DeepClusteringLoss(nn.Module):
    def __init__(self):
        """
        Computes the deep clustering loss with weights. Equation (7) in [1].

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

        norm = 1. / (((((weights) ** 2)).sum(dim=1) ** 2).sum() + 1e-10)

        assignments = weights.expand_as(assignments) * assignments
        embedding = weights.expand_as(embedding) * embedding

        vTv = ((embedding.transpose(2, 1) @ embedding) ** 2).sum()
        vTy = ((embedding.transpose(2, 1) @ assignments) ** 2).sum()
        yTy = ((assignments.transpose(2, 1) @ assignments) ** 2).sum()
        loss = (vTv - 2 * vTy + yTy)
        return loss * norm

class PermutationInvariantLoss(nn.Module):
    def __init__(self, loss_function):
        """Computes the Permutation Invariant Loss (PIT) [1] by permuting the estimated 
        sources and the reference sources. Takes the best permutation and only backprops
        the loss from that.
        [1] Yu, Dong, Morten Kolbæk, Zheng-Hua Tan, and Jesper Jensen. 
            "Permutation invariant training of deep models for speaker-independent 
            multi-talker speech separation." In 2017 IEEE International Conference on 
            Acoustics, Speech and Signal Processing (ICASSP), pp. 241-245. IEEE, 2017.
        """
        super(PermutationInvariantLoss, self).__init__()
        self.loss_function = loss_function
        self.loss_function.reduction = 'none'
        
    def forward(self, estimates, targets):
        num_batch = estimates.shape[0]
        num_sources = estimates.shape[-1]
        estimates = estimates.view(num_batch, -1, num_sources)
        targets = targets.view(num_batch, -1, num_sources)
        
        losses = []
        for p in permutations(range(num_sources)):
            loss = self.loss_function(estimates[:, :, list(p)], targets)
            loss = loss.mean(dim=1).mean(dim=-1)
            losses.append(loss)
        
        losses = torch.stack(losses,dim=-1)
        losses = torch.min(losses, dim=-1)[0]
        loss = torch.mean(losses)
        return loss

class CombinationInvariantLoss(nn.Module):
    def __init__(self, loss_function):
        """Variant on Permutation Invariant Loss where instead a combination of the
        sources output by the model are used. This way a model can output more 
        sources than there are in the ground truth. A subset of the output sources
        will be compared using Permutation Invariant Loss with the ground truth
        estimates.
        """
        super(CombinationInvariantLoss, self).__init__()
        self.loss_function = loss_function
        self.loss_function.reduction = 'none'
        
    def forward(self, estimates, targets):
        num_batch = estimates.shape[0]
        num_target_sources = targets.shape[-1]
        num_estimate_sources = estimates.shape[-1]

        estimates = estimates.view(num_batch, -1, num_estimate_sources)
        targets = targets.view(num_batch, -1, num_target_sources)
        
        losses = []
        for c in combinations(range(num_estimate_sources), num_target_sources):
            _estimates = estimates[:, :, list(c)]
            for p in permutations(range(num_target_sources)):
                loss = self.loss_function(_estimates[:, :, list(p)], targets)
                loss = loss.mean(dim=1).mean(dim=-1)
                losses.append(loss)
        
        losses = torch.stack(losses,dim=-1)
        losses = torch.min(losses, dim=-1)[0]
        loss = torch.mean(losses)
        return loss