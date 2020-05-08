import sys
import builtins
import nussl
import os
import numpy as np
import pytest
import importlib


def test_jupyter_embed_audio(benchmark_audio):
    for key, path in benchmark_audio.items():
        s1 = nussl.AudioSignal(path)
        audio_element = nussl.play_utils.embed_audio(s1)
        assert os.path.splitext(audio_element.filename)[-1] == '.mp3'

        audio_element = nussl.play_utils.embed_audio(s1, ext='.wav')
        assert os.path.splitext(audio_element.filename)[-1] == '.wav'

        s1.embed_audio()


def test_jupyter_no_ffmpy(benchmark_audio, monkeypatch):
    import_orig = builtins.__import__

    def mocked_import(name, globals_, locals_, fromlist, level):
        if name == 'ffmpy':
            raise ImportError()
        return import_orig(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', mocked_import)

    for key, path in benchmark_audio.items():
        s1 = nussl.AudioSignal(path)
        audio_element = nussl.play_utils.embed_audio(s1)
        assert os.path.splitext(audio_element.filename)[-1] == '.wav'

    monkeypatch.undo()


def test_jupyter_no_ipython(benchmark_audio, monkeypatch):
    import_orig = builtins.__import__

    def mocked_import(name, globals_, locals_, fromlist, level):
        if name == 'IPython':
            raise ImportError()
        return import_orig(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', mocked_import)

    for key, path in benchmark_audio.items():
        s1 = nussl.AudioSignal(path)
        pytest.raises(ImportError, nussl.play_utils.embed_audio, s1)

    monkeypatch.undo()

def test_musdb_fail_to_import(monkeypatch):
    import_orig = builtins.__import__

    def mocked_import(name, globals_, locals_, fromlist, level):
        if name == 'musdb':
            raise RuntimeError()
        return import_orig(name, globals_, locals_, fromlist, level)

    def reimport_nussl():
        importlib.reload(nussl)

    monkeypatch.setattr(builtins, '__import__', mocked_import)
    pytest.warns(UserWarning, reimport_nussl)
    monkeypatch.undo()

def test_play_audio():
    audio_signal = nussl.AudioSignal(
        audio_data_array=np.zeros(100), sample_rate=1000)
    nussl.play_utils.play(audio_signal)
    audio_signal.play()


def test_multitrack():
    nussl.utils.seed(0)

    audio_path = nussl.efz_utils.download_audio_file(
        'marimba_timbre.mp3')
    audio_signal = nussl.AudioSignal(audio_path, duration=1)
    separator = nussl.separation.primitive.TimbreClustering(
        audio_signal, 2, 1, mask_type='binary')

    estimates = separator()

    html = nussl.play_utils.multitrack(
        estimates, ['Cluster 0', 'Cluster 1'], display=False)

    os.makedirs('tests/core/regression/', exist_ok=True)
    regression_path = 'tests/core/regression/multitrack_test.html'

    with open(regression_path, 'w') as f:
        f.write(html.data)

    # once more for coverage
    html = nussl.play_utils.multitrack(
        estimates, ['Cluster 0', 'Cluster 1'], display=True)

    # names are optional
    html = nussl.play_utils.multitrack(estimates, display=False)

    # can take a dictionary
    _estimates = {i: e for i, e in enumerate(estimates)}
    html = nussl.play_utils.multitrack(_estimates, display=False)

    # names should match length
    pytest.raises(ValueError,
                  nussl.play_utils.multitrack, estimates, ['not enough names'])
