from nussl import separation, datasets, AudioSignal, evaluation
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

    assert separator.sample_rate == mix.sample_rate
    assert separator.stft_params == mix.stft_params

    pytest.raises(NotImplementedError, separator.run)
    pytest.raises(NotImplementedError, separator.make_audio_signals)
    pytest.raises(NotImplementedError, separator.get_metadata)
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


    separator = separation.SeparationBase(mix)
    assert separator.audio_signal == mix

    monkeypatch.setattr(separation.SeparationBase, 'make_audio_signals', dummy_run)

    separator(audio_signal=sources['s1'])
    assert separator.audio_signal == sources['s1']

def test_separation_base_interact(mix_source_folder, monkeypatch):
    dataset = datasets.MixSourceFolder(mix_source_folder)
    item = dataset[0]
    mix = item['mix']

    def dummy_run(self):
        return self.audio_signal

    class DummyGradio():
        def __init__(*args, **kwargs):
            pass
        def launch(self, *args, **kwargs):
            pass

    import gradio
    
    monkeypatch.setattr(separation.SeparationBase, 'make_audio_signals', dummy_run)
    monkeypatch.setattr(separation.SeparationBase, 'run', dummy_run)
    monkeypatch.setattr(gradio, 'Interface', DummyGradio)

    separator = separation.SeparationBase(mix)
    separator.interact()
    separator.interact(add_residual=True)


def test_clustering_separation_base(scaper_folder, monkeypatch):
    dataset = datasets.Scaper(scaper_folder)
    item = dataset[5]
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

        pytest.raises(SeparationException, separator.confidence)

        estimates = separator()
        confidence = separator.confidence()
        assert confidence == 1.0

        evaluator = evaluation.BSSEvalScale(
            list(sources.values()), estimates, compute_permutation=True)
        scores = evaluator.evaluate()

        for key in evaluator.source_labels:
            for metric in ['SI-SDR', 'SI-SIR']:
                _score = scores[key][metric]  
                for val in _score:
                    assert val > 5

        separator = separation.ClusteringSeparationBase(
            mix, 2, clustering_type=clustering_type, mask_type='binary')

        estimates = separator()

        evaluator = evaluation.BSSEvalScale(
            list(sources.values()), estimates, compute_permutation=True)
        scores = evaluator.evaluate()

        for key in evaluator.source_labels:
            for metric in ['SI-SDR', 'SI-SIR']:
                _score = scores[key][metric]  
                for val in _score:
                    assert val > 9  
