from itertools import permutations, combinations

import torch
import torch.nn as nn

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
    ``nussl.evaluation.bss_eval.scale_bss_eval`` and can be used to compute
    SI-SDR or SNR.
    
    Args:
        scaling (bool, optional): Whether to use scale-invariant (True) or
          signal-to-noise ratio (False). Defaults to True.
        return_scaling (bool, optional): Whether to only return the scaling
          factor that the estimate gets scaled by relative to the reference. 
          This is just for monitoring this value during training, don't actually
          train with it! Defaults to False.
        reduction (str, optional): How to reduce across the batch (either 'mean', 
          'sum', or none). Defaults to 'mean'.
        zero_mean (bool, optional): Zero mean the references and estimates before
          computing the loss. Defaults to True.
    """
    DEFAULT_KEYS = {'audio': 'estimates', 'source_audio': 'references'}

    def __init__(self, scaling=True, return_scaling=False, reduction='mean', 
                 zero_mean=True):
        self.scaling = scaling
        self.reduction = reduction
        self.zero_mean = zero_mean
        self.return_scaling = return_scaling
        super().__init__()

    def forward(self, estimates, references):
        eps = 1e-8
        # num_batch, num_samples, num_sources
        _shape = references.shape
        references = references.reshape(-1, _shape[-2], _shape[-1])
        estimates = estimates.reshape(-1, _shape[-2], _shape[-1])
        # samples now on axis 1
        if self.zero_mean:
            mean_reference = references.mean(dim=1, keepdim=True)
            mean_estimate = estimates.mean(dim=1, keepdim=True)
        else:
            mean_reference = 0
            mean_estimate = 0
        
        _references = references - mean_reference
        _estimates = estimates - mean_estimate
        
        references_projection = (_references ** 2).sum(dim=-2) + eps
        references_on_estimates = (_estimates * _references).sum(dim=-2) + eps

        scale = (
            (references_on_estimates / references_projection).unsqueeze(1)
            if self.scaling else 1)

        e_true = scale * _references
        e_res = _estimates - e_true

        signal = (e_true ** 2).sum(dim=1)
        noise = (e_res ** 2).sum(dim=1)
        sdr = 10 * torch.log10(signal / noise + eps)        

        if self.reduction == 'mean':
            sdr = sdr.mean()
        elif self.reduction == 'sum':
            sdr = sdr.sum()
        if self.return_scaling:
            return scale
        # go negative so it's a loss
        return -sdr


class DeepClusteringLoss(nn.Module):
    """
    Computes the deep clustering loss with weights. Equation (7) in [1].

    References:

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

        weights = weights.view(batch_size, -1, 1)

        # make everything unit norm
        embedding = embedding.reshape(batch_size, -1, embedding_size)
        embedding = nn.functional.normalize(embedding, dim=-1, p=2)

        assignments = assignments.view(batch_size, -1, num_sources)
        assignments = nn.functional.normalize(assignments, dim=-1, p=2)

        norm = (((weights.reshape(batch_size, -1)) ** 2).sum(dim=1) ** 2) + 1e-8

        assignments = weights * assignments
        embedding = weights * embedding

        vTv = ((embedding.transpose(2, 1) @ embedding) ** 2).reshape(
            batch_size, -1).sum(dim=-1)
        vTy = ((embedding.transpose(2, 1) @ assignments) ** 2).reshape(
            batch_size, -1).sum(dim=-1)
        yTy = ((assignments.transpose(2, 1) @ assignments) ** 2).reshape(
            batch_size, -1).sum(dim=-1)
        loss = (vTv - 2 * vTy + yTy) / norm
        return loss.mean()

class WhitenedKMeansLoss(nn.Module):
    """
    Computes the whitened K-Means loss with weights. Equation (6) in [1].

    References:

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
        super(WhitenedKMeansLoss, self).__init__()

    def forward(self, embedding, assignments, weights):
        batch_size = embedding.shape[0]
        embedding_size = embedding.shape[-1]
        num_sources = assignments.shape[-1]

        weights = weights.view(batch_size, -1, 1)

        # make everything unit norm
        embedding = embedding.reshape(batch_size, -1, embedding_size)
        embedding = nn.functional.normalize(embedding, dim=-1, p=2)

        assignments = assignments.view(batch_size, -1, num_sources)
        assignments = nn.functional.normalize(assignments, dim=-1, p=2)

        assignments = weights * assignments
        embedding = weights * embedding

        embedding_dim_identity = torch.eye(
            embedding_size, device=embedding.device).float()
        source_dim_identity = torch.eye(
            num_sources, device=embedding.device).float()

        vTv = (embedding.transpose(2, 1) @ embedding)
        vTy = (embedding.transpose(2, 1) @ assignments)
        yTy = (assignments.transpose(2, 1) @ assignments)

        ivTv = torch.inverse(vTv + embedding_dim_identity)
        iyTy = torch.inverse(yTy + source_dim_identity)

        ivTv_vTy = ivTv @ vTy
        vTy_iyTy = vTy @ iyTy

        # tr(AB) = sum(A^T o B) 
        # where o denotes element-wise product
        # this is the trace trick
        # http://andreweckford.blogspot.com/2009/09/trace-tricks.html
        trace = (ivTv_vTy * vTy_iyTy).sum()
        D = (embedding_size + num_sources) * batch_size
        loss = D - 2 * trace
        return loss / batch_size

class PermutationInvariantLoss(nn.Module):
    """
    Computes the Permutation Invariant Loss (PIT) [1] by permuting the estimated 
    sources and the reference sources. Takes the best permutation and only backprops
    the loss from that.

    For when you're trying to match the estimates to the sources but you don't 
    know the order in which your model outputs the estimates.

    References:
    
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
        estimates = estimates.reshape(num_batch, -1, num_sources)
        targets = targets.reshape(num_batch, -1, num_sources)
        
        losses = []
        for p in permutations(range(num_sources)):
            _targets = targets[..., list(p)]
            loss = self.loss_function(estimates, _targets)
            loss = loss.mean(dim=-1)
            losses.append(loss)
        
        losses = torch.stack(losses, dim=-1)
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

    For when you're trying to match the estimates to the sources but you don't 
    know the order in which your model outputs the estimates AND you are 
    outputting more estimates then there are sources.

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
        
        estimates = estimates.reshape(num_batch, -1, num_estimate_sources)
        targets = targets.reshape(num_batch, -1, num_target_sources)
        
        losses = []
        for c in combinations(range(num_estimate_sources), num_target_sources):
            _estimates = estimates[..., list(c)]
            for p in permutations(range(num_target_sources)):
                _targets = targets[..., list(p)]
                loss = self.loss_function(_estimates, _targets)
                loss = loss.mean(dim=-1)
                losses.append(loss)
        
        losses = torch.stack(losses, dim=-1)
        losses = torch.min(losses, dim=-1)[0]
        loss = torch.mean(losses)
        return loss
