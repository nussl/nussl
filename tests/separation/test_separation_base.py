from nussl import separation, datasets, AudioSignal, core
import pytest 
import numpy as np

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
    assert separator.mask_type == separation.MaskSeparationBase.SOFT_MASK
    assert separator.mask_threshold == 0.5

    separator = separation.MaskSeparationBase(mix, 
        mask_type=core.masks.SoftMask(mask_shape=(100, 10)))
    assert separator.mask_type == separation.MaskSeparationBase.SOFT_MASK

    separator = separation.MaskSeparationBase(mix, mask_type='binary')
    assert separator.mask_type == separation.MaskSeparationBase.BINARY_MASK

    separator = separation.MaskSeparationBase(mix, 
        mask_type=core.masks.BinaryMask(mask_shape=(100, 10)))
    assert separator.mask_type == separation.MaskSeparationBase.BINARY_MASK

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
    masked = mix.apply_mask(zeros_mask)
    masked.istft()

    assert np.allclose(masked.audio_data, np.zeros(masked.audio_data.shape), 
        atol=1e-6)

    separator = separation.MaskSeparationBase(mix, mask_type='binary')
    zeros_mask = separator.zeros_mask(mix.stft().shape)
    masked = mix.apply_mask(zeros_mask)
    masked.istft()

    assert np.allclose(masked.audio_data, np.zeros(masked.audio_data.shape), 
        atol=1e-6)