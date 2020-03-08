from nussl import separation, datasets, AudioSignal
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
