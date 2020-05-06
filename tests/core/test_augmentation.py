import numpy as np
import numpy.random as random
from nussl.core.augmentation import *
from nussl.core.augmentation_utils import *
import nussl.datasets.hooks as hooks
import tempfile
import pytest
import os
import os.path as path
import librosa
import pytest

REGRESSION_PATH="tests/core/regression/augmentation"
os.makedirs(REGRESSION_PATH, exist_ok=True)

# # We test on a single item from MUSDB
# musdb = hooks.MUSDB18(download=True)
# item = musdb[40]

# # Put sources in tempfiles
# item_tempfile = tempfile.NamedTemporaryFile()
# mix_loc = item_tempfile.name
# item["mix"].write_audio_to_file(item_tempfile.name)
# source_tempfiles = {}
# for name, source in item["sources"].items():
#     source_tempfile = tempfile.NamedTemporaryFile()
#     source.write_audio_to_file(source_tempfile.name)
#     source_tempfiles[name] = source_tempfile


def test_stretch(mix_and_sources):
    stretch_factor = .8
    mix, _ = mix_and_sources
    augmented = time_stretch(mix, stretch_factor)

    assert np.allclose(augmented.audio_data[0, :], 
            librosa.effects.time_stretch(np.asfortranarray(mix.audio_data[0, :]), stretch_factor))

def test_pitch_shift(mix_and_sources):
    shift = 2
    mix, _ = mix_and_sources
    sample_rate = mix.sample_rate

    augmented = pitch_shift(mix, shift)

    assert np.allclose(augmented.audio_data[0, :], 
            librosa.effects.pitch_shift(np.asfortranarray(mix.audio_data[0, :]), sample_rate, shift))

def test_params(mix_and_sources):
    audio_signal, _ = mix_and_sources

    with pytest.raises(ValueError): 
        time_stretch(audio_signal, -1)
    with pytest.raises(ValueError):
        time_stretch(audio_signal, "not scalar")

    with pytest.raises(ValueError):
        pitch_shift(audio_signal, 1.4)

    with pytest.raises(ValueError):
        pitch_shift(audio_signal, "this is a string")

def ffmpeg_regression(audio_data, reg_path, check_against_regression_data):
    scores = {
        'test_metrics': {
            "mean": [np.mean(audio_data, dtype=float)],
            "t_0_mean": [np.mean(audio_data[:, 0], dtype=float)],
            "variance": [np.var(audio_data, dtype=float)],
            "f_0_mean": [np.mean(audio_data[0, :], dtype=float)],
            "sum": [np.sum(audio_data, dtype=float)]
        }
    }
    check_against_regression_data(scores, reg_path)

def test_tremolo(mix_and_sources, check_against_regression_data):
    f = 15
    d = .5
    mix, _ =  mix_and_sources
    reg_path = path.join(REGRESSION_PATH, "tremolo.json")

    augmented_signal = tremolo(mix, f, d)

    ffmpeg_regression(augmented_signal.audio_data, 
        reg_path, check_against_regression_data)

def test_vibrato(mix_and_sources, check_against_regression_data):
    f = 5
    d = .5
    reg_path = path.join(REGRESSION_PATH, "vibrato.json")
    mix, _ =  mix_and_sources
    augmented_signal = vibrato(mix, f, d)
    
    ffmpeg_regression(augmented_signal.audio_data, 
        reg_path, check_against_regression_data)
    
def test_echo(mix_and_sources, check_against_regression_data):
    # This sounds like an open air concert in the mountains
    in_gain = .8
    out_gain = .9
    delays = [1000]
    decays = [.3]
    mix, _ = mix_and_sources

    reg_path = path.join(REGRESSION_PATH, "echo.json")
    echo_mix = echo(mix, in_gain, out_gain, delays, decays)
    ffmpeg_regression(echo_mix.audio_data, 
        reg_path, check_against_regression_data)

    # Same as above but with one more mountain
    delays = [1000, 1800]
    decays = [.3 , .25]
    reg_path = path.join(REGRESSION_PATH, "echo2.json")
    echo_mix = echo(mix, in_gain, out_gain, delays, decays)
    ffmpeg_regression(echo_mix.audio_data, 
        reg_path, check_against_regression_data)

    