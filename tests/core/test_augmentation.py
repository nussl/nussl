import numpy as np
import numpy.random as random
import nussl.core.augmentation as augment
from nussl.core.augmentation_utils import apply_ffmpeg_filter
import nussl.datasets.hooks as hooks
import tempfile
import pytest
import os
import os.path as path
import librosa
import pytest

REGRESSION_PATH="tests/core/regression/augmentation"
os.makedirs(REGRESSION_PATH, exist_ok=True)

def test_stretch(mix_and_sources):
    stretch_factor = .8
    mix, _ = mix_and_sources
    augmented = augment.time_stretch(mix, stretch_factor)

    assert np.allclose(augmented.audio_data[0, :], 
            librosa.effects.time_stretch(np.asfortranarray(mix.audio_data[0, :]), stretch_factor))

def test_pitch_shift(mix_and_sources):
    shift = 2
    mix, _ = mix_and_sources
    sample_rate = mix.sample_rate

    augmented = augment.pitch_shift(mix, shift)

    assert np.allclose(augmented.audio_data[0, :], 
            librosa.effects.pitch_shift(np.asfortranarray(mix.audio_data[0, :]), sample_rate, shift))

def test_params(mix_and_sources):
    audio_signal, _ = mix_and_sources

    with pytest.raises(ValueError): 
        augment.time_stretch(audio_signal, -1)
    with pytest.raises(ValueError):
        augment.time_stretch(audio_signal, "not scalar")

    with pytest.raises(ValueError):
        augment.pitch_shift(audio_signal, 1.4)

    with pytest.raises(ValueError):
        augment.pitch_shift(audio_signal, "this is a string")

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

    augmented_signal = augment.tremolo(mix, f, d)

    ffmpeg_regression(augmented_signal.audio_data, 
        reg_path, check_against_regression_data)

def test_vibrato(mix_and_sources, check_against_regression_data):
    f = 5
    d = .5
    reg_path = path.join(REGRESSION_PATH, "vibrato.json")
    mix, _ =  mix_and_sources
    augmented_signal = augment.vibrato(mix, f, d)
    
    ffmpeg_regression(augmented_signal.audio_data, 
        reg_path, check_against_regression_data)
    
# def test_echo(mix_and_sources, check_against_regression_data):
#     # This sounds like an open air concert in the mountains
#     in_gain = .8
#     out_gain = .9
#     delays = [1000]
#     decays = [.3]
#     mix, _ = mix_and_sources

#     reg_path = path.join(REGRESSION_PATH, "echo.json")
#     echo_mix = echo(mix, in_gain, out_gain, delays, decays)
#     ffmpeg_regression(echo_mix.audio_data, 
#         reg_path, check_against_regression_data)

#     # Same as above but with one more mountain
#     delays = [1000, 1800]
#     decays = [.3 , .25]
#     reg_path = path.join(REGRESSION_PATH, "echo2.json")
#     echo_mix = echo(mix, in_gain, out_gain, delays, decays)
#     ffmpeg_regression(echo_mix.audio_data, 
#         reg_path, check_against_regression_data)

def test_emphasis(mix_and_sources, check_against_regression_data):
    mix, _  = mix_and_sources
    level_in = 1
    level_out = 2
    types = ['col', 'riaa', 'cd']
    mode = 'production'

    for _type in types:
        reg_path = path.join(REGRESSION_PATH, f"emphasis_{_type}.json")
        augmented_signal = augment.emphasis(mix, level_in, level_out, _type, mode=mode)
        ffmpeg_regression(augmented_signal.audio_data, reg_path, check_against_regression_data)

def test_compressor(mix_and_sources, check_against_regression_data):
    mix, _ = mix_and_sources
    level_in = 1

    reg_path = path.join(REGRESSION_PATH, f"compression.json")
    augmented_signal = augment.compressor(mix, level_in)
    ffmpeg_regression(augmented_signal.audio_data, reg_path, check_against_regression_data)

def test_compression_fail(mix_and_sources):
    signal, _ = mix_and_sources

    # Out of bounds values
    level_in = -1
    reduction_ratio = 299
    attack = 19999
    release = 89999
    makeup = 69
    knee = 6
    link = "fail"
    detection = "fail"
    mix = 8
    threshold = 0

    with pytest.raises(ValueError):
        augment.compressor(signal, level_in)

    with pytest.raises(ValueError):
        augment.compressor(signal, 1, reduction_ratio=reduction_ratio)
    
    with pytest.raises(ValueError):
        augment.compressor(signal, 1, attack=attack)
    
    with pytest.raises(ValueError):
        augment.compressor(signal, 1, release=release)
    
    with pytest.raises(ValueError):
        augment.compressor(signal, 1, makeup=makeup)
    
    with pytest.raises(ValueError):
        augment.compressor(signal, 1, link=link)
    
    with pytest.raises(ValueError):
        augment.compressor(signal, 1, detection=detection)
    
    with pytest.raises(ValueError):
        augment.compressor(signal, 1, attack=attack)

    with pytest.raises(ValueError):
        augment.compressor(signal, 1, mix=mix)

    with pytest.raises(ValueError):
        augment.compressor(signal, 1, threshold=threshold)

def test_equalizer(mix_and_sources, check_against_regression_data):
    bands = [
        {
            "chn": [0, 1],
            "f": i,
            "w": (5 * np.log2(i)),
            "g": 5,
            "t": 0
        } for i in [110, 440, 10000]
    ]

    signal, _ = mix_and_sources
    augmented_signal = augment.equalizer(signal, bands)
    
    reg_path = path.join(REGRESSION_PATH, "equalizer.json")
    ffmpeg_regression(augmented_signal.audio_data, 
        reg_path, check_against_regression_data)

def test_phaser(mix_and_sources, check_against_regression_data):
    mix, _ = mix_and_sources

    in_gain = .4
    out_gain = .74
    delay = 3
    decay = .4
    speed = .8

    augmented_signal = augment.phaser(mix, in_gain, out_gain, delay, speed)
    reg_path = path.join(REGRESSION_PATH, "phaser.json")
    ffmpeg_regression(augmented_signal.audio_data, reg_path,
        check_against_regression_data)
    
def test_chorus(mix_and_sources, check_against_regression_data):
    mix, _ = mix_and_sources

    in_gain = .4
    out_gain = .7
    delays = [45, 55]
    decays = [.6, .2]
    speeds = [.9, .8]
    depths = [2, 1.5]

    augmented_signal = augment.chorus(mix, delays, decays,
        speeds, depths, in_gain=in_gain, out_gain=out_gain)
    reg_path = path.join(REGRESSION_PATH, "chorus.json")
    ffmpeg_regression(augmented_signal.audio_data, reg_path,
        check_against_regression_data)

def test_flanger(mix_and_sources, check_against_regression_data):
    mix, _ = mix_and_sources

    delay = 10
    depth = 2
    regen = 10
    width = 70
    speed = .5
    phase = 25

    augmented_signal = augment.flanger(mix, delay, depth, regen, width,
        speed, phase)
    reg_path = path.join(REGRESSION_PATH, "flanger.json")
    ffmpeg_regression(augmented_signal.audio_data, reg_path,
        check_against_regression_data)

def test_low_high_pass(mix_and_sources):
    mix, _  = mix_and_sources
    f = 512
    idx = 16
    low_mix = augment.low_pass(mix, f)
    mix.stft_data = None
    high_mix = augment.high_pass(mix, f)
    l_stft = low_mix.stft_data
    h_stft = high_mix.stft_data
    assert np.allclose(l_stft[16:], np.zeros_like(l_stft[16:]))
    assert np.allclose(h_stft[:15], np.zeros_like(h_stft[:15]))

    with pytest.raises(ValueError):
        augment.low_pass(mix, -1)
    with pytest.raises(ValueError):
        augment.low_pass(mix, "fail")
    
    with pytest.raises(ValueError):
        augment.high_pass(mix, -1)
    with pytest.raises(ValueError):
        augment.high_pass(mix, "fail")

def test_misc_param_check(mix_and_sources):
    signal, _ = mix_and_sources
    with pytest.raises(ValueError):
        augment.vibrato(signal, -1, 1)
    with pytest.raises(ValueError):
        augment.vibrato(signal, 1, -1)
    
    with pytest.raises(ValueError):
        augment.tremolo(signal, -1, 1)
    with pytest.raises(ValueError):
        augment.tremolo(signal, 1, -1)

    with pytest.raises(ValueError):
        augment.chorus(signal, [1], [2,3],[45,6],[4])  
    with pytest.warns(UserWarning):
        augment.chorus(signal, [2000],[.5], [.7], [.4])
    with pytest.raises(ValueError):
        augment.chorus(signal, [1],[30], [.7], [.4], in_gain=-1)

    with pytest.raises(ValueError):
        augment.phaser(signal, in_gain=-1)
    with pytest.raises(ValueError):
        augment.phaser(signal, _type="fail")

    with pytest.raises(ValueError):
        augment.flanger(signal, delay=-1)
    with pytest.raises(ValueError):
        augment.flanger(signal, depth=-1)
    with pytest.raises(ValueError):
        augment.flanger(signal, regen=-100)
    with pytest.raises(ValueError):
        augment.flanger(signal, width=-1)
    with pytest.raises(ValueError):
        augment.flanger(signal, speed=-1)
    with pytest.raises(ValueError):
        augment.flanger(signal, shape="fail")
    with pytest.raises(ValueError):
        augment.flanger(signal, interp="fail")
    
    with pytest.raises(ValueError):
        augment.emphasis(signal, 1, 1, _type="fail")
    with pytest.raises(ValueError):
        augment.emphasis(signal, 1, 1, mode="fail")
    with pytest.raises(ValueError):
        augment.emphasis(signal, 1, -1)

    band = {
        'chn':[0],
        'f': 300,
        'w': 10,
        'g': 10,
        't': 5
    }
    with pytest.raises(ValueError):
        augment.equalizer(signal, [band])
    band["g"] = -1
    with pytest.raises(ValueError):
        augment.equalizer(signal, [band])
    band["w"] = -1
    with pytest.raises(ValueError):
        augment.equalizer(signal, [band])
    band["f"] = -1
    with pytest.raises(ValueError):
        augment.equalizer(signal, [band])
    band["chn"].append(100)
    with pytest.raises(ValueError):
        augment.equalizer(signal, [band])
    

def test_filter_debug_mode(mix_and_sources):
    signal, _ = mix_and_sources
    apply_ffmpeg_filter(signal, "tremolo", silent=False)