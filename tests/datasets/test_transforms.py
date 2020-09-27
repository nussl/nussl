import pytest
from nussl.datasets import transforms
from nussl.datasets.transforms import TransformException
import nussl
from nussl import STFTParams, evaluation
import numpy as np
from nussl.core.masks import BinaryMask, SoftMask
import itertools
import copy
import torch
import tempfile
import os

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

    sources = [sources[k] for k in sources]
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
        'sources': sources
    }

    msa = transforms.MagnitudeSpectrumApproximation()
    assert isinstance(str(msa), str)
    psa = transforms.PhaseSensitiveSpectrumApproximation()
    assert isinstance(str(psa), str)

    assert msa.__class__.__name__ in str(msa)
    assert psa.__class__.__name__ in str(psa)

    pytest.raises(TransformException, psa, {'sources': 'blah'})
    pytest.raises(TransformException, msa, {'sources': 'blah'})

    _data = {'mix': mix}
    output = msa(_data)
    assert np.allclose(output['mix_magnitude'], np.abs(mix.stft()))

    output = msa(data)
    assert np.allclose(output['mix_magnitude'], np.abs(mix.stft()))
    assert list(data['sources'].keys()) == sorted(list(sources.keys()))

    masks = []
    estimates = []

    shape = mix.stft_data.shape + (len(sources),)

    mix_masks = np.ones(shape)
    mix_scores = separate_and_evaluate(mix, data['sources'], mix_masks)

    ibm_scores = separate_and_evaluate(
        mix, data['sources'], data['ideal_binary_mask'])
    output['source_magnitudes'] += 1e-8

    mask_data = (
            output['source_magnitudes'] /
            np.maximum(
                output['mix_magnitude'][..., None],
                output['source_magnitudes'])
    )
    msa_scores = separate_and_evaluate(mix, data['sources'], mask_data)

    _data = {'mix': mix}
    output = psa(_data)
    assert np.allclose(output['mix_magnitude'], np.abs(mix.stft()))

    output = psa(data)
    assert np.allclose(output['mix_magnitude'], np.abs(mix.stft()))
    assert list(data['sources'].keys()) == sorted(list(sources.keys()))

    output['source_magnitudes'] += 1e-8

    mask_data = (
            output['source_magnitudes'] /
            np.maximum(
                output['mix_magnitude'][..., None],
                output['source_magnitudes'])
    )
    psa_scores = separate_and_evaluate(mix, data['sources'], mask_data)

    for key in msa_scores:
        if key in ['SI-SDR', 'SI-SIR', 'SI-SAR']:
            diff = np.array(psa_scores[key]) - np.array(mix_scores[key])
            assert diff.mean() > 10


def test_transform_sum_sources(musdb_tracks):
    track = musdb_tracks[10]
    mix, sources = nussl.utils.musdb_track_to_audio_signals(track)

    data = {
        'mix': mix,
        'sources': sources
    }

    groups = itertools.combinations(data['sources'].keys(), 3)

    tfm = None
    for group in groups:
        _data = copy.deepcopy(data)
        tfm = transforms.SumSources([group])
        assert isinstance(str(tfm), str)
        _data = tfm(_data)
        for g in group:
            assert g not in _data['sources']
        assert '+'.join(group) in _data['sources']

        summed_sources = sum([sources[k] for k in group])

        assert np.allclose(
            _data['sources']['+'.join(group)].audio_data,
            summed_sources.audio_data
        )

    pytest.raises(TransformException, tfm, {'no_key'})

    pytest.raises(TransformException,
                  transforms.SumSources, 'test')

    pytest.raises(TransformException,
                  transforms.SumSources,
                  [['vocals', 'test'], ['test2', 'test3']],
                  ['mygroup']
                  )


def test_transform_compose(musdb_tracks):
    track = musdb_tracks[10]
    mix, sources = nussl.utils.musdb_track_to_audio_signals(track)

    data = {
        'mix': mix,
        'sources': sources,
        'metadata': {
            'labels': ['bass', 'drums', 'other', 'vocals']
        }
    }

    class _BadTransform(object):
        def __call__(self, data):
            return 'not a dictionary'

    com = transforms.Compose([_BadTransform()])
    pytest.raises(TransformException, com, data)

    msa = transforms.MagnitudeSpectrumApproximation()
    tfm = transforms.SumSources(
        [['other', 'drums', 'bass']],
        group_names=['accompaniment']
    )
    assert isinstance(str(tfm), str)
    com = transforms.Compose([tfm, msa])
    assert msa.__class__.__name__ in str(com)
    assert tfm.__class__.__name__ in str(com)

    data = com(data)

    assert np.allclose(data['mix_magnitude'], np.abs(mix.stft()))
    assert data['metadata']['labels'] == [
        'bass', 'drums', 'other', 'vocals', 'accompaniment']

    mask_data = (
            data['source_magnitudes'] /
            np.maximum(
                data['mix_magnitude'][..., None],
                data['source_magnitudes'])
    )
    msa_scores = separate_and_evaluate(mix, data['sources'], mask_data)
    shape = mix.stft_data.shape + (len(sources),)
    mask_data = np.ones(shape)
    mix_scores = separate_and_evaluate(mix, data['sources'], mask_data)

    for key in msa_scores:
        if key in ['SI-SDR', 'SI-SIR', 'SI-SAR']:
            diff = np.array(msa_scores[key]) - np.array(mix_scores[key])
            assert diff.mean() > 10


def test_transform_to_separation_model(musdb_tracks):
    track = musdb_tracks[10]
    mix, sources = nussl.utils.musdb_track_to_audio_signals(track)

    data = {
        'mix': mix,
        'sources': sources,
        'metadata': {'labels': []}
    }

    msa = transforms.MagnitudeSpectrumApproximation()
    tdl = transforms.ToSeparationModel()
    assert tdl.__class__.__name__ in str(tdl)

    com = transforms.Compose([msa, tdl])

    data = com(data)
    accepted_keys = ['mix_magnitude', 'source_magnitudes']
    rejected_keys = ['mix', 'sources', 'metadata']

    for a in accepted_keys:
        assert a in data
    for r in rejected_keys:
        assert r not in data

    for key in data:
        assert torch.is_tensor(data[key])
        assert data[key].shape[0] == mix.stft().shape[1]
        assert data[key].shape[1] == mix.stft().shape[0]


def test_transform_get_excerpt(musdb_tracks):
    track = musdb_tracks[10]
    mix, sources = nussl.utils.musdb_track_to_audio_signals(track)

    msa = transforms.MagnitudeSpectrumApproximation()
    tdl = transforms.ToSeparationModel()
    excerpt_lengths = [400, 1000, 2000]
    for excerpt_length in excerpt_lengths:
        data = {
            'mix': mix,
            'sources': sources,
            'metadata': {'labels': []}
        }

        exc = transforms.GetExcerpt(excerpt_length=excerpt_length)
        assert isinstance(str(exc), str)
        com = transforms.Compose([msa, tdl, exc])

        data = com(data)

        for key in data:
            assert torch.is_tensor(data[key])
            assert data[key].shape[0] == excerpt_length
            assert data[key].shape[1] == mix.stft().shape[0]

        assert torch.mean((data['source_magnitudes'].sum(dim=-1) -
                           data['mix_magnitude']) ** 2).item() < 1e-5

        data = {
            'mix': mix,
            'sources': sources,
            'metadata': {'labels': []}
        }

        exc = transforms.GetExcerpt(excerpt_length=excerpt_length)
        assert isinstance(str(exc), str)
        com = transforms.Compose([msa, tdl])

        data = com(data)
        for key in data:
            data[key] = data[key].cpu().data.numpy()

        data = exc(data)

        for key in data:
            assert data[key].shape[0] == excerpt_length
            assert data[key].shape[1] == mix.stft().shape[0]

        assert np.mean((data['source_magnitudes'].sum(axis=-1) -
                        data['mix_magnitude']) ** 2) < 1e-5

        data = {
            'mix_magnitude': 'not an array or tensor'
        }

        pytest.raises(TransformException, exc, data)
    
    excerpt_lengths = [1009, 16000, 612140]
    ga = transforms.GetAudio()
    for excerpt_length in excerpt_lengths:
        data = {
            'mix': sum(sources.values()),
            'sources': sources,
            'metadata': {'labels': []}
        }

        exc = transforms.GetExcerpt(
            excerpt_length=excerpt_length,
            tf_keys = ['mix_audio', 'source_audio'],
            time_dim=1,
        )
        com = transforms.Compose([ga, tdl, exc])

        data = com(data)

        for key in data:
            assert torch.is_tensor(data[key])
            assert data[key].shape[1] == excerpt_length

        assert torch.allclose(
            data['source_audio'].sum(dim=-1), data['mix_audio'], atol=1e-3)


def test_transform_cache(musdb_tracks):
    track = musdb_tracks[10]
    mix, sources = nussl.utils.musdb_track_to_audio_signals(track)

    data = {
        'mix': mix,
        'sources': sources,
        'metadata': {'labels': sorted(list(sources.keys()))},
        'index': 0
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tfm = transforms.Cache(
            os.path.join(tmpdir, 'cache'), cache_size=2, overwrite=True)

        _data_a = tfm(data)
        _info_a = tfm.info

        tfm.overwrite = False

        _data_b = tfm({'index': 0})

        pytest.raises(TransformException, tfm, {})
        pytest.raises(TransformException, tfm, {'index': 1})

        for key in _data_a:
            assert _data_a[key] == _data_b[key]

        com = transforms.Compose([
            transforms.MagnitudeSpectrumApproximation(),
            transforms.ToSeparationModel(),
            transforms.Cache(
                os.path.join(tmpdir, 'cache'),
                overwrite=True),
        ])

        _data_a = com(data)
        com.transforms[-1].overwrite = False
        _data_b = com.transforms[-1]({'index': 0})

        for key in _data_a:
            if torch.is_tensor(_data_a[key]):
                assert torch.allclose(_data_a[key], _data_b[key])
            else:
                assert _data_a[key] == _data_b[key]


def test_transforms_labels_to_one_hot(mix_source_folder, scaper_folder):
    dataset = nussl.datasets.MixSourceFolder(mix_source_folder)
    item = dataset[0]

    tfm = transforms.LabelsToOneHot()
    assert isinstance(str(tfm), str)

    one_hots = tfm(item)['one_hot_labels']
    assert np.allclose(one_hots, np.eye(2))

    item['sources'].pop('s0')
    one_hots = tfm(item)['one_hot_labels']
    assert np.allclose(one_hots, np.array([0, 1]))

    dataset = nussl.datasets.Scaper(scaper_folder)
    item = dataset[0]

    one_hots = tfm(item)['one_hot_labels']
    assert one_hots.shape[-1] == len(item['metadata']['labels'])

    item['metadata'].pop('labels')
    pytest.raises(TransformException, tfm, item)

    item.pop('metadata')
    pytest.raises(TransformException, tfm, item)


def test_transforms_magnitude_weights(mix_source_folder):
    dataset = nussl.datasets.MixSourceFolder(mix_source_folder)
    item = dataset[0]

    tfm = transforms.MagnitudeWeights()
    assert isinstance(str(tfm), str)
    pytest.raises(TransformException, tfm, {'sources': []})

    item_from_mix = tfm(item)

    msa = transforms.MagnitudeSpectrumApproximation()
    item = tfm(msa(item))

    assert item['weights'].shape == item['mix_magnitude'].shape
    assert np.allclose(item_from_mix['weights'], item['weights'])


def test_transforms_index_sources(mix_source_folder):
    dataset = nussl.datasets.MixSourceFolder(mix_source_folder)
    item = dataset[0]

    index = 1
    tfm = transforms.IndexSources('source_magnitudes', index)
    assert isinstance(str(tfm), str)

    pytest.raises(TransformException, tfm, {'sources': []})
    pytest.raises(TransformException, tfm,
                  {'source_magnitudes': np.random.randn(100, 100, 1)})

    msa = transforms.MagnitudeSpectrumApproximation()
    msa_output = copy.deepcopy(msa(item))

    item = tfm(msa(item))

    assert (
        np.allclose(
            item['source_magnitudes'],
            msa_output['source_magnitudes'][..., index, None])
    )

def test_transform_get_audio(mix_source_folder):
    dataset = nussl.datasets.MixSourceFolder(mix_source_folder)
    item = dataset[0]

    index = 1
    tfm = transforms.GetAudio()
    assert isinstance(str(tfm), str)
    pytest.raises(TransformException, tfm, {'sources': []})

    ga_output = tfm(item)

    assert np.allclose(
        ga_output['mix_audio'], item['mix'].audio_data)
    source_names = sorted(list(item['sources'].keys()))

    for i, key in enumerate(source_names):
        assert np.allclose(
            ga_output['source_audio'][..., i], item['sources'][key].audio_data)

    item.pop('sources')
    item.pop('source_audio')

    ga_output = tfm(item)

    assert np.allclose(
        ga_output['mix_audio'], item['mix'].audio_data)

