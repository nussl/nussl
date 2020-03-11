from nussl import separation, datasets, AudioSignal, core, evaluation
import pytest 
import numpy as np
from nussl.separation.base import SeparationException

def test_separation_base(mix_source_folder, monkeypatch):
    dataset = datasets.MixSourceFolder(mix_source_folder)
    item = dataset[0]
    mix = item['mix']
    sources = item['sources']

    pytest.warns(UserWarning, separation.SeparationBase, AudioSignal())
    pytest.raises(ValueError, separation.SeparationBase, None)

    separator = separation.SeparationBase(mix)
    separator.plot() # shouldn't error out

    assert separator.sample_rate == mix.sample_rate
    assert separator.stft_params == mix.stft_params

    pytest.raises(NotImplementedError, separator.run)
    pytest.raises(NotImplementedError, separator.make_audio_signals)
    pytest.raises(NotImplementedError, separator)

    def dummy_run(self):
        pass

    monkeypatch.setattr(separation.SeparationBase, 'run', dummy_run)
    pytest.raises(NotImplementedError, separator)

    assert separator.__class__.__name__ in str(separator)
    assert str(mix) in str(separator)

    other = separation.SeparationBase(mix)
    separator.fake_array = np.zeros(100)
    other.fake_array = np.zeros(100)

    assert separator == other

    other.fake_array = np.ones(100)
    assert separator != other

    diff_other = separation.SeparationBase(sources['s1'])
    diff_other.fake_array = np.zeros(100)

    assert separator != diff_other

def test_mask_separation_base(mix_source_folder, monkeypatch):
    dataset = datasets.MixSourceFolder(mix_source_folder)
    item = dataset[0]
    mix = item['mix']
    sources = item['sources']

    class DummyMask(core.masks.MaskBase):
        @staticmethod
        def _validate_mask(mask_):
            pass
        pass

    separator = separation.MaskSeparationBase(mix)
    assert separator.mask_type == core.masks.SoftMask
    assert separator.mask_threshold == 0.5

    separator = separation.MaskSeparationBase(mix, 
        mask_type=core.masks.SoftMask(mask_shape=(100, 10)))
    assert separator.mask_type == core.masks.SoftMask

    separator = separation.MaskSeparationBase(mix, mask_type='binary')
    assert separator.mask_type == core.masks.BinaryMask

    separator = separation.MaskSeparationBase(mix, 
        mask_type=core.masks.BinaryMask(mask_shape=(100, 10)))
    assert separator.mask_type == core.masks.BinaryMask

    pytest.raises(ValueError, separation.MaskSeparationBase, mix, mask_type=None)
    pytest.raises(ValueError, separation.MaskSeparationBase, mix, 
        mask_type='invalid')
    pytest.raises(ValueError, separation.MaskSeparationBase, mix, 
        mask_type=DummyMask(mask_shape=(100, 10)))

    separator = separation.MaskSeparationBase(mix, mask_threshold=0.2)
    assert separator.mask_threshold == 0.2

    pytest.raises(ValueError, separation.MaskSeparationBase, mix, 
        mask_threshold=1.5)
    pytest.raises(ValueError, separation.MaskSeparationBase, mix, 
        mask_threshold='not a float')

    separator = separation.MaskSeparationBase(mix)
    ones_mask = separator.ones_mask(mix.stft().shape)
    masked = mix.apply_mask(ones_mask)
    masked.istft()

    assert np.allclose(masked.audio_data, mix.audio_data, atol=1e-6)

    separator = separation.MaskSeparationBase(mix, mask_type='binary')
    ones_mask = separator.ones_mask(mix.stft().shape)
    masked = mix.apply_mask(ones_mask)
    masked.istft()

    assert np.allclose(masked.audio_data, mix.audio_data, atol=1e-6)

    separator = separation.MaskSeparationBase(mix)
    zeros_mask = separator.zeros_mask(mix.stft().shape)
    masked_zeros = mix.apply_mask(zeros_mask)
    masked_zeros.istft()

    assert np.allclose(masked_zeros.audio_data, np.zeros(masked_zeros.audio_data.shape), 
        atol=1e-6)

    separator = separation.MaskSeparationBase(mix, mask_type='binary')
    ones_mask = separator.ones_mask(mix.stft().shape)
    zeros_mask = separator.zeros_mask(mix.stft().shape)
    masked_ones = mix.apply_mask(ones_mask)
    masked_ones.istft()

    assert np.allclose(masked_ones.audio_data, mix.audio_data, atol=1e-6)

    pytest.raises(SeparationException, separator.make_audio_signals)

    separator = separation.MaskSeparationBase(mix, mask_type='binary')
    separator.result_masks = [ones_mask, zeros_mask]
    estimates = separator.make_audio_signals()

    for e, s in zip(estimates, [masked_ones, masked_zeros]):
        assert e == s

    separator = separation.MaskSeparationBase(mix, mask_type='soft')
    separator.result_masks = [ones_mask, zeros_mask]

    pytest.raises(SeparationException, separator.make_audio_signals)

def test_clustering_separation_base(mix_source_folder, monkeypatch):
    dataset = datasets.MixSourceFolder(mix_source_folder)
    item = dataset[0]
    mix = item['mix']
    sources = item['sources']

    pytest.raises(SeparationException, separation.ClusteringSeparationBase, 
        mix, 2, clustering_type='not allowed')

    clustering_types = (
        separation.base.clustering_separation_base.ALLOWED_CLUSTERING_TYPES)

    separator = separation.ClusteringSeparationBase(mix, 2)
    bad_features = np.ones(100)
    pytest.raises(SeparationException, separator.run, bad_features)

    good_features = np.stack([
            np.abs(s.stft()) for _, s in sources.items()
        ], axis=-1)

    good_features = (
        good_features == good_features.max(axis=-1, keepdims=True))

    def dummy_extract(self):
        return good_features

    monkeypatch.setattr(
        separation.ClusteringSeparationBase, 'extract_features', dummy_extract)

    for clustering_type in clustering_types:
        separator = separation.ClusteringSeparationBase(
            mix, 2, clustering_type=clustering_type)

        estimates = separator()

        evaluator = evaluation.BSSEvalScale(
            list(sources.values()), estimates, compute_permutation=True)
        scores = evaluator.evaluate()

        for key in evaluator.source_labels:
            for metric in ['SDR', 'SIR']:
                _score = scores[key][metric]  
                for val in _score:
                    assert val > 9    

        separator = separation.ClusteringSeparationBase(
            mix, 2, clustering_type=clustering_type, mask_type='binary')

        estimates = separator()

        evaluator = evaluation.BSSEvalScale(
            list(sources.values()), estimates, compute_permutation=True)
        scores = evaluator.evaluate()

        for key in evaluator.source_labels:
            for metric in ['SDR', 'SIR']:
                _score = scores[key][metric]  
                for val in _score:
                    assert val > 9  
