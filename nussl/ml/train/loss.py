import torch
import torch.nn as nn
import numpy as np
from itertools import permutations, combinations
from enum import Enum

class L1Loss(nn.L1Loss):
    DEFAULT_KEYS = {'estimates': 'input', 'source_magnitudes': 'target'}

class MSELoss(nn.MSELoss):
    DEFAULT_KEYS = {'estimates': 'input', 'source_magnitudes': 'target'}

class KLDivLoss(nn.KLDivLoss):
    DEFAULT_KEYS = {'estimates': 'input', 'source_magnitudes': 'target'}

class SISDRLoss(nn.Module):
    """
    Computes the Scale-Invariant Source-to-Distortion Ratio between a batch
    of estimated and reference audio signals. Used in end-to-end networks.
    This is essentially a batch PyTorch version of the function 
    ``nussl.evaluation.bss_eval.scale_bss_eval``.
    
    Args:
        scaling (bool, optional): Whether to use scale-invariant (True) or
          scale-dependent SDR. Defaults to True.
    """
    DEFAULT_KEYS = {'estimates': 'estimates', 'references': 'references'}

    def __init__(self, scaling=True, reduction='mean'):
        self.scaling = scaling
        self.reduction = reduction
        super().__init__()

    def forward(self, estimates, references):
        references_projection = references.transpose(2, 1) @ references

        references_projection = torch.diagonal(
            references_projection, dim1=-2, dim2=-1)
    
        references_on_estimates = torch.diagonal(
            references.transpose(2, 1) @ estimates, dim1=-2, dim2=-1)

        scale = (
            (references_on_estimates / references_projection).unsqueeze(1)
            if self.scaling else 1)

        e_true = scale * references
        e_res = estimates - e_true

        signal = (e_true ** 2).sum(dim=1)
        noise = (e_res ** 2).sum(dim=1)

        SDR = 10 * torch.log10(signal / noise)

        if self.reduction == 'mean':
            SDR = SDR.mean()
        elif self.reduction == 'sum':
            SDR = SDR.sum()
        # go negative so it's a loss
        return -SDR

class DeepClusteringLoss(nn.Module):
    """
    Computes the deep clustering loss with weights. Equation (7) in [1].

    [1] Wang, Z. Q., Le Roux, J., & Hershey, J. R. (2018, April).
    Alternative Objective Functions for Deep Clustering.
    In Proc. IEEE International Conference on Acoustics,  Speech
    and Signal Processing (ICASSP).
    """
    DEFAULT_KEYS = {
        'embedding': 'embedding', 
        'ideal_binary_mask': 'assignments', 
        'weights': 'weights'
    }

    def __init__(self):
        super(DeepClusteringLoss, self).__init__()

    def forward(self, embedding, assignments, weights):
        batch_size = embedding.shape[0]
        embedding_size = embedding.shape[-1]
        num_sources = assignments.shape[-1]

        eps = 1e-12

        weights = weights.view(batch_size, -1, 1)

        # make data unit norm without affecting gradient
        embedding = embedding.view(batch_size, -1, embedding_size)

        denom = embedding.norm(2, dim=-1, keepdim=True).clamp_min(
            eps).expand_as(embedding)
        embedding = embedding / denom.data

        assignments = assignments.view(batch_size, -1, num_sources)

        denom = assignments.norm(2, dim=-1, keepdim=True).clamp_min(
            eps).expand_as(assignments)
        assignments = assignments / denom.data

        assignments = weights * assignments
        embedding = weights * embedding

        vTv = ((embedding.transpose(2, 1) @ embedding) ** 2).reshape(
            batch_size, -1).sum(dim=-1)
        vTy = ((embedding.transpose(2, 1) @ assignments) ** 2).reshape(
            batch_size, -1).sum(dim=-1)
        yTy = ((assignments.transpose(2, 1) @ assignments) ** 2).reshape(
            batch_size, -1).sum(dim=-1)
        loss = (vTv - 2 * vTy + yTy) / (vTv + yTy).detach()
        return loss.mean()

class PermutationInvariantLoss(nn.Module):
    """
    Computes the Permutation Invariant Loss (PIT) [1] by permuting the estimated 
    sources and the reference sources. Takes the best permutation and only backprops
    the loss from that.
    
    [1] Yu, Dong, Morten Kolbæk, Zheng-Hua Tan, and Jesper Jensen. 
    "Permutation invariant training of deep models for speaker-independent 
    multi-talker speech separation." In 2017 IEEE International Conference on 
    Acoustics, Speech and Signal Processing (ICASSP), pp. 241-245. IEEE, 2017.
    """
    DEFAULT_KEYS = {'estimates': 'estimates', 'source_magnitudes': 'targets'}

    def __init__(self, loss_function):
        
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
    """
    Variant on Permutation Invariant Loss where instead a combination of the
    sources output by the model are used. This way a model can output more 
    sources than there are in the ground truth. A subset of the output sources
    will be compared using Permutation Invariant Loss with the ground truth
    estimates.
    """
    DEFAULT_KEYS = {'estimates': 'estimates', 'source_magnitudes': 'targets'}

    def __init__(self, loss_function):
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
