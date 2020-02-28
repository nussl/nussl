import pytest
from nussl.datasets import transforms
from nussl.datasets.transforms import TransformException
import nussl
from nussl import STFTParams, evaluation
import numpy as np
from nussl.core.masks import BinaryMask, SoftMask

stft_tol = 1e-6

def separate_and_evaluate(mix, sources, mask_data):
    estimates = []
    mask_data = normalize_masks(mask_data)
    for i in range(mask_data.shape[-1]):
        mask = SoftMask(mask_data[..., i])
        estimate = mix.apply_mask(mask)
        estimate.istft()
        estimates.append(estimate)

    assert np.allclose(
        sum(estimates).audio_data, mix.audio_data, atol=stft_tol)

    evaluator = evaluation.BSSEvalScale(
        sources, estimates)
    scores = evaluator.evaluate()
    return scores

def normalize_masks(mask_data):
    mask_data = (
        mask_data / 
        np.sum(mask_data, axis=-1, keepdims=True) + 1e-8
    )
    return mask_data

def test_transform_msa_psa(musdb_tracks):
    track = musdb_tracks[10]
    mix, sources = nussl.utils.musdb_track_to_audio_signals(track)

    data = {
        'mix': mix,
        'sources': [sources[k] for k in sources]
    }

    msa = transforms.MagnitudeSpectrumApproximation()
    psa = transforms.PhaseSensitiveSpectrumApproximation()

    pytest.raises(TransformException, psa, {'mix': 'blah'})
    pytest.raises(TransformException, msa, {'mix': 'blah'})

    output = msa(data)
    assert np.allclose(output['mix_magnitude'], np.abs(mix.stft()))

    masks = []
    estimates = []

    shape = mix.stft_data.shape + (len(sources),)

    mix_masks = np.ones(shape)
    mix_scores = separate_and_evaluate(mix, data['sources'], mix_masks)

    ibm_scores = separate_and_evaluate(mix, data['sources'], data['ideal_binary_mask'])
    output['source_magnitudes'] += 1e-8

    mask_data = (
        output['source_magnitudes'] / 
        np.maximum(
            output['mix_magnitude'][..., None], 
            output['source_magnitudes'])
    )
    msa_scores = separate_and_evaluate(mix, data['sources'], mask_data)

    output = psa(data)
    assert np.allclose(output['mix_magnitude'], np.abs(mix.stft()))
    output['source_magnitudes'] += 1e-8

    mask_data = (
        output['source_magnitudes'] / 
        np.maximum(
            output['mix_magnitude'][..., None], 
            output['source_magnitudes'])
    )
    psa_scores = separate_and_evaluate(mix, data['sources'], mask_data)

    for key in msa_scores:
        if key in ['SDR', 'SIR', 'SAR']:
            diff = np.array(psa_scores[key]) - np.array(mix_scores[key])
            assert diff.mean() > 10
