import sys
import builtins
import nussl
import os
import numpy as np
import pytest

def test_jupyter_embed_audio(benchmark_audio):
    for key, path in benchmark_audio.items():
        s1 = nussl.AudioSignal(path)
        audio_element = nussl.play_utils.embed_audio(s1)
        assert os.path.splitext(audio_element.filename)[-1] == '.mp3'

        audio_element = nussl.play_utils.embed_audio(s1, ext='.wav')
        assert os.path.splitext(audio_element.filename)[-1] == '.wav'


def test_jupyter_no_ffmpy(benchmark_audio, monkeypatch):
    import_orig = builtins.__import__
    def mocked_import(name, globals, locals, fromlist, level):
        if name == 'ffmpy':
            raise ImportError()
        return import_orig(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, '__import__', mocked_import)

    for key, path in benchmark_audio.items():
        s1 = nussl.AudioSignal(path)
        audio_element = nussl.play_utils.embed_audio(s1)
        assert os.path.splitext(audio_element.filename)[-1] == '.wav'

    monkeypatch.undo()

def test_jupyter_no_ipython(benchmark_audio, monkeypatch):
    import_orig = builtins.__import__
    def mocked_import(name, globals, locals, fromlist, level):
        if name == 'IPython':
            raise ImportError()
        return import_orig(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, '__import__', mocked_import)

    for key, path in benchmark_audio.items():
        s1 = nussl.AudioSignal(path)
        pytest.raises(ImportError, nussl.play_utils.embed_audio, s1)
    
    monkeypatch.undo()

def test_play_audio():
    audio_signal = nussl.AudioSignal(
        audio_data_array=np.zeros(100), sample_rate=1000)
    nussl.play_utils.play(audio_signal)
    