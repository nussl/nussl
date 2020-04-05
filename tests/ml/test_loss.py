import torch
import nussl
from nussl import ml
from torch import nn
import numpy as np
from itertools import permutations
import random


def test_register_loss():
    class ExampleLoss(nn.Module):
        DEFAULT_KEYS = {'key1': 'arg1', 'key2': 'arg2'}

        def forward(self, arg1, arg2):
            return 0

    assert ExampleLoss.__name__ not in (dir(ml.train.loss))
    ml.register_loss(ExampleLoss)
    assert ExampleLoss.__name__ in (dir(ml.train.loss))


def test_deep_clustering_loss():
    n_batch = 40
    n_time = 400
    n_freq = 129
    n_sources = 4
    n_embedding = 20

    embedding = torch.rand(n_batch, n_time, n_freq, n_embedding)
    embedding = torch.nn.functional.normalize(
        embedding, dim=-1, p=2)

    assignments = torch.rand(n_batch, n_time, n_freq, n_sources) > .5
    assignments = assignments.float()

    weights = torch.ones(n_batch, n_time, n_freq)

    LossDPCL = ml.train.loss.DeepClusteringLoss()
    _loss_a = LossDPCL(assignments, assignments, weights).item()
    assert _loss_a == 0

    _loss_b = LossDPCL(embedding, assignments, weights).item()
    assert _loss_b > _loss_a
    assert _loss_b <= 1


def test_permutation_invariant_loss_tf():
    n_batch = 10
    n_time = 400
    n_freq = 129
    n_sources = 4

    sources = torch.randn(n_batch, n_time, n_freq, n_sources)

    LossPIT = ml.train.loss.PermutationInvariantLoss(
        loss_function=ml.train.loss.L1Loss())
    LossL1 = ml.train.loss.L1Loss()
    noise_amount = [0.001, .01, .05, 1.0]

    for n in noise_amount:
        permuted = []
        noisy = sources + n * torch.randn_like(sources)
        _loss_a = LossL1(noisy, sources).item()

        for i in range(n_batch):
            p = random.choice(list(permutations(range(n_sources))))
            permuted_batch = noisy[i, ..., list(p)].unsqueeze(0)
            permuted.append(permuted_batch)

        permuted = torch.cat(permuted, dim=0)
        _loss_b = LossPIT(permuted, sources).item()
        assert np.allclose(_loss_a, _loss_b, atol=1e-6)


def test_combination_invariant_loss_tf():
    n_batch = 40
    n_time = 400
    n_freq = 129
    n_sources = 2

    sources = torch.randn(n_batch, n_time, n_freq, n_sources)

    LossCPIT = ml.train.loss.CombinationInvariantLoss(
        loss_function=nn.L1Loss())
    LossL1 = nn.L1Loss()

    _loss_a = LossL1(sources, sources).item()

    for shift in range(n_sources):
        sources_a = sources[:, :, :, shift:]
        sources_b = sources[:, :, :, :shift]
        sources_c = torch.randn(n_batch, n_time, n_freq, n_sources)
        shifted_sources = torch.cat(
            [sources_a, sources_b, sources_c], dim=-1)
        _loss_b = LossCPIT(shifted_sources, sources).item()
        assert _loss_a == _loss_b


def test_sdr_loss():
    n_batch = 40
    n_samples = 16000
    n_sources = 2

    references = torch.randn(n_batch, n_samples, n_sources)

    noise_amount = [0.01, 0.05, 0.1, 0.2, 0.5, 1]
    LossSDR = ml.train.loss.SISDRLoss()
    prev_loss = -np.inf

    for n in noise_amount:
        estimates = references + n * torch.randn(n_batch, n_samples, n_sources)
        _loss = LossSDR(estimates, references).item()
        assert _loss > prev_loss
        prev_loss = _loss

    LossSDR = ml.train.loss.SISDRLoss(reduction='none')
    prev_loss = -np.inf

    for n in noise_amount:
        estimates = references + n * torch.randn(n_batch, n_samples, n_sources)
        _loss = LossSDR(estimates, references)
        assert _loss.sum().item() > prev_loss
        prev_loss = _loss.sum().item()

        idx = np.random.randint(n_batch)
        _numpy_si_sdr = nussl.evaluation.scale_bss_eval(
            references.data.numpy()[idx], 
            estimates.data.numpy()[idx, ..., 0],
            references.data.numpy()[idx].sum(axis=-1),
            0
        )[0]

        _torch_loss_on_one = -1 * _loss[idx][0]

        assert np.allclose(_torch_loss_on_one.item(), _numpy_si_sdr, atol=1e-3)

def test_permutation_invariant_loss_sdr():
    n_batch = 40
    n_samples = 16000
    n_sources = 2

    references = torch.randn(n_batch, n_samples, n_sources)

    noise_amount = [0.01, 0.05, 0.1, 0.5, 1.0]
    LossPIT = ml.train.loss.PermutationInvariantLoss(
        loss_function=ml.train.loss.SISDRLoss())
    LossSDR = ml.train.loss.SISDRLoss()

    for n in noise_amount:
        estimates = references + n * torch.randn(n_batch, n_samples, n_sources)
        _loss_a = LossSDR(estimates, references).item()

        for shift in range(n_sources):
            sources_a = estimates[..., shift:]
            sources_b = estimates[..., :shift]
            shifted_sources = torch.cat(
                [sources_a, sources_b], dim=-1)
            _loss_b = LossPIT(shifted_sources, references).item()
            assert np.allclose(_loss_a, _loss_b, atol=1e-4)


def test_combination_invariant_loss_sdr():
    n_batch = 40
    n_samples = 16000
    n_sources = 2

    references = torch.randn(n_batch, n_samples, n_sources)

    noise_amount = [0.01, 0.05, 0.1, 0.5, 1.0]
    LossCPIT = ml.train.loss.CombinationInvariantLoss(
        loss_function=ml.train.loss.SISDRLoss())
    LossSDR = ml.train.loss.SISDRLoss()

    for n in noise_amount:
        estimates = references + n * torch.randn(n_batch, n_samples, n_sources)
        _loss_a = LossSDR(estimates, references).item()

        for shift in range(n_sources):
            sources_a = estimates[..., shift:]
            sources_b = estimates[..., :shift]
            sources_c = torch.rand_like(estimates)
            shifted_sources = torch.cat(
                [sources_a, sources_b, sources_c], dim=-1)
            _loss_b = LossCPIT(shifted_sources, references).item()
            assert np.allclose(_loss_a, _loss_b, atol=1e-4)
