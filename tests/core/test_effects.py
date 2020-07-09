import numpy as np
import nussl.core.effects as effects
from nussl.core.audio_signal import AudioSignalException
import os
import os.path as path
import pytest

REGRESSION_PATH = "tests/core/regression/augmentation"
os.makedirs(REGRESSION_PATH, exist_ok=True)


def fx_regression(audio_data, reg_path, check_against_regression_data):
    scores = {
        'test_metrics': {
            "variance": [np.sum(np.var(audio_data, dtype=float, axis=1))],
            "sum": np.sum(audio_data, dtype=float, axis=1).tolist()
        }
    }
    check_against_regression_data(scores, reg_path)


def test_stretch(mix_and_sources, check_against_regression_data):
    stretch_factor = .8
    signal, _ = mix_and_sources
    filters = [effects.time_stretch(stretch_factor)]
    augmented = effects.apply_effects_sox(signal, filters)

    reg_path = path.join(REGRESSION_PATH, "time_stretch.json")
    fx_regression(augmented.audio_data, reg_path, check_against_regression_data)


def test_pitch_shift(mix_and_sources, check_against_regression_data):
    shift = 2
    signal, _ = mix_and_sources
    filters = [effects.pitch_shift(shift)]
    augmented = effects.apply_effects_sox(signal, filters)

    reg_path = path.join(REGRESSION_PATH, "pitch_shift.json")
    fx_regression(augmented.audio_data, reg_path, check_against_regression_data)


def test_params(mix_and_sources):
    with pytest.raises(ValueError):
        effects.time_stretch(-1)

    with pytest.raises(ValueError):
        effects.time_stretch("not numeric")

    with pytest.raises(ValueError):
        effects.pitch_shift(1.4)

    with pytest.raises(ValueError):
        effects.pitch_shift("this is a string")


def test_tremolo(mix_and_sources, check_against_regression_data):
    f = 15
    d = .5
    signal, _ = mix_and_sources
    reg_path = path.join(REGRESSION_PATH, "tremolo.json")

    filters = [effects.tremolo(f, d)]
    augmented_signal = effects.apply_effects_ffmpeg(signal, filters)

    fx_regression(augmented_signal.audio_data,
                  reg_path, check_against_regression_data)


def test_vibrato(mix_and_sources, check_against_regression_data):
    f = 5
    d = .5
    reg_path = path.join(REGRESSION_PATH, "vibrato.json")
    signal, _ = mix_and_sources

    filters = [effects.vibrato(f, d)]
    augmented_signal = effects.apply_effects_ffmpeg(signal, filters)

def test_emphasis(mix_and_sources, check_against_regression_data):
    signal, _ = mix_and_sources
    level_in = 1
    level_out = 2
    types = ['col', 'riaa', 'cd']
    mode = 'production'

    for _type in types:
        reg_path = path.join(REGRESSION_PATH, f"emphasis_{_type}.json")
        filters = [effects.emphasis(level_in, level_out, _type, mode=mode)]
        augmented_signal = effects.apply_effects_ffmpeg(signal, filters)
        fx_regression(augmented_signal.audio_data, reg_path, check_against_regression_data)


def test_compressor(mix_and_sources, check_against_regression_data):
    signal, _ = mix_and_sources
    level_in = 1

    reg_path = path.join(REGRESSION_PATH, f"compression.json")
    filters = [effects.compressor(level_in)]
    augmented_signal = effects.apply_effects_ffmpeg(signal, filters)
    fx_regression(augmented_signal.audio_data, reg_path, check_against_regression_data)


# noinspection DuplicatedCode
def test_compression_fail(mix_and_sources):
    # Out of bounds values
    level_in = -1
    reduction_ratio = 299
    attack = 19999
    release = 89999
    makeup = 69
    knee = 10
    link = "fail"
    detection = "fail"
    mode = "fail"
    mix = 8
    threshold = 0
    phase = -1

    with pytest.raises(ValueError):
        effects.compressor(level_in)

    with pytest.raises(ValueError):
        effects.compressor(1, knee=knee)

    with pytest.raises(ValueError):
        effects.compressor(1, reduction_ratio=reduction_ratio)

    with pytest.raises(ValueError):
        effects.compressor(1, release=release)

    with pytest.raises(ValueError):
        effects.compressor(1, makeup=makeup)

    with pytest.raises(ValueError):
        effects.compressor(1, link=link)

    with pytest.raises(ValueError):
        effects.compressor(1, detection=detection)

    with pytest.raises(ValueError):
        effects.compressor(1, attack=attack)

    with pytest.raises(ValueError):
        effects.compressor(1, mix=mix)

    with pytest.raises(ValueError):
        effects.compressor(1, threshold=threshold)

    with pytest.raises(ValueError):
        effects.compressor(1, mode=mode)

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
    filters = [effects.equalizer(bands)]
    augmented_signal = effects.apply_effects_ffmpeg(signal, filters)

    reg_path = path.join(REGRESSION_PATH, "equalizer.json")
    fx_regression(augmented_signal.audio_data,
                  reg_path, check_against_regression_data)


def test_phaser(mix_and_sources, check_against_regression_data):
    signal, _ = mix_and_sources

    in_gain = .4
    out_gain = .74
    delay = 3
    speed = .8

    filters = [effects.phaser(in_gain, out_gain, delay, speed)]
    augmented_signal = effects.apply_effects_ffmpeg(signal, filters)
    reg_path = path.join(REGRESSION_PATH, "phaser.json")
    fx_regression(augmented_signal.audio_data, reg_path,
                  check_against_regression_data)


def test_chorus(mix_and_sources, check_against_regression_data):
    signal, _ = mix_and_sources

    in_gain = .4
    out_gain = .7
    delays = [45, 55]
    decays = [.6, .2]
    speeds = [.9, .8]
    depths = [2, 1.5]

    filters = [effects.chorus(delays, decays,
                              speeds, depths, in_gain=in_gain, out_gain=out_gain)]
    augmented_signal = effects.apply_effects_ffmpeg(signal, filters)
    reg_path = path.join(REGRESSION_PATH, "chorus.json")
    fx_regression(augmented_signal.audio_data, reg_path,
                  check_against_regression_data)


def test_flanger(mix_and_sources, check_against_regression_data):
    signal, _ = mix_and_sources

    delay = 10
    depth = 2
    regen = 10
    width = 70
    speed = .5
    phase = 25

    filters = [effects.flanger(delay=delay, depth=depth, regen=regen,
                               width=width, speed=speed, phase=phase)]
    augmented_signal = effects.apply_effects_ffmpeg(signal, filters)
    reg_path = path.join(REGRESSION_PATH, "flanger.json")
    fx_regression(augmented_signal.audio_data, reg_path,
                  check_against_regression_data)


def test_low_high_pass(mix_and_sources, check_against_regression_data):
    signal, _ = mix_and_sources
    f = 512
    filters = [effects.low_pass(f)]
    low_signal = effects.apply_effects_ffmpeg(signal, filters)
    low_path = path.join(REGRESSION_PATH, "low_pass.json")
    fx_regression(low_signal.audio_data, low_path,
                  check_against_regression_data)

    filters = [effects.high_pass(f)]
    high_signal = effects.apply_effects_ffmpeg(signal, filters)
    high_path = path.join(REGRESSION_PATH, "high_pass.json")
    fx_regression(high_signal.audio_data, high_path,
                  check_against_regression_data)

    with pytest.raises(ValueError):
        effects.low_pass(-1)
    with pytest.raises(ValueError):
        effects.low_pass("fail")
    with pytest.raises(ValueError):
        effects.low_pass(1, poles=0)
    with pytest.raises(ValueError):
        effects.low_pass(1, width_type=0)
    with pytest.raises(ValueError):
        effects.low_pass(1, width=-1)

    with pytest.raises(ValueError):
        effects.high_pass(-1)
    with pytest.raises(ValueError):
        effects.high_pass("fail")
    with pytest.raises(ValueError):
        effects.high_pass(1, poles=0)
    with pytest.raises(ValueError):
        effects.high_pass(1, width_type=0)
    with pytest.raises(ValueError):
        effects.high_pass(1, width=-1)


def test_misc_param_check():
    with pytest.raises(ValueError):
        effects.vibrato(-1, 1)
    with pytest.raises(ValueError):
        effects.vibrato(1, -1)

    with pytest.raises(ValueError):
        effects.tremolo(-1, 1)
    with pytest.raises(ValueError):
        effects.tremolo(1, -1)

    with pytest.raises(ValueError):
        effects.chorus([1], [2, 3], [45, 6], [4])
    with pytest.warns(UserWarning):
        effects.chorus([2000], [.5], [.7], [.4])
    with pytest.raises(ValueError):
        effects.chorus([1], [30], [.7], [.4], in_gain=-1)

    with pytest.raises(ValueError):
        effects.phaser(in_gain=-1)
    with pytest.raises(ValueError):
        effects.phaser(out_gain=-1)
    with pytest.raises(ValueError):
        effects.phaser(decay=-1)
    with pytest.raises(ValueError):
        effects.phaser(_type="fail")

    with pytest.raises(ValueError):
        effects.flanger(delay=-1)
    with pytest.raises(ValueError):
        effects.flanger(depth=-1)
    with pytest.raises(ValueError):
        effects.flanger(regen=-100)
    with pytest.raises(ValueError):
        effects.flanger(width=-1)
    with pytest.raises(ValueError):
        effects.flanger(speed=-1)
    with pytest.raises(ValueError):
        effects.flanger(shape="fail")
    with pytest.raises(ValueError):
        effects.flanger(interp="fail")
    with pytest.raises(ValueError):
        effects.flanger(phase=phase)

    with pytest.raises(ValueError):
        effects.emphasis(1, 1, _type="fail")
    with pytest.raises(ValueError):
        effects.emphasis(1, 1, mode="fail")
    with pytest.raises(ValueError):
        effects.emphasis(1, -1)

    band = {
        'chn': [0],
        'f': 300,
        'w': 10,
        'g': 10,
        't': 5
    }
    with pytest.raises(ValueError):
        effects.equalizer([band])
    band["g"] = -1
    with pytest.raises(ValueError):
        effects.equalizer([band])
    band["w"] = -1
    with pytest.raises(ValueError):
        effects.equalizer([band])
    band["f"] = -1
    with pytest.raises(ValueError):
        effects.equalizer([band])
    band["chn"].append(-1)
    with pytest.raises(ValueError):
        effects.equalizer([band])

    with pytest.raises(ValueError):
        effects.SoXFilter('fail')


def test_silent_mode(mix_and_sources):
    signal, _ = mix_and_sources
    effects.apply_effects_ffmpeg(signal, [], silent=True)


def test_hooks(mix_and_sources, check_against_regression_data):
    signal, _ = mix_and_sources

    signal = (
        signal
            .time_stretch(3)
            .pitch_shift(2)
            .low_pass(512)
            .high_pass(512)
            .tremolo(5, .4)
            .vibrato(3, .9)
            .chorus([20, 70], [.9, .4], [.9, .6], [1, .9])
            .phaser()
            .flanger(delay=3)
            .emphasis(1, .5, _type='riaa')
            .compressor(.9)
            .equalizer([{
                'chn': [0, 1],
                'f': 512,
                'w': 10,
                'g': 4,
                't': 0
            }])
    )

    assert len(signal._ffmpeg_effects_chain) == 10
    assert len(signal._sox_effects_chain) == 2

    augmented_signal = signal.apply_effects(reset=False)

    reg_path = path.join(REGRESSION_PATH, "hooks.json")
    fx_regression(augmented_signal.audio_data, reg_path, check_against_regression_data)

    augmented_signal.time_stretch(.7).apply_effects(overwrite=True)

    assert len(augmented_signal._ffmpeg_effects_chain) == 0
    assert len(augmented_signal._sox_effects_chain) == 0


def test_make_effect(mix_and_sources, check_against_regression_data):
    signal, _ = mix_and_sources
    signal.reset_effects_chain()

    (signal
    .make_effect("time_stretch", factor=3)
    .make_effect("pitch_shift", shift=2)
    .make_effect("low_pass", freq=512)
    .make_effect("high_pass", freq=512)
    .make_effect("tremolo", mod_freq=5, mod_depth=.4)
    .make_effect("vibrato", mod_freq=3, mod_depth=.9)
    .make_effect("chorus", delays=[20, 70], decays=[.9, .4], speeds=[.9, .6], depths=[1, .9])
    .make_effect("phaser")
    .make_effect("flanger", delay=3)
    .make_effect("emphasis", level_in=1, level_out=.5, _type='riaa')
    .make_effect("compressor", level_in=.9)
    .make_effect("equalizer", bands=[{
                'chn': [0, 1],
                'f': 512,
                'w': 10,
                'g': 4,
                't': 0
            }])
    )
    signal.apply_effects(reset=False)
    augmented_signal = signal.apply_effects(reset=False, overwrite=True)
    reg_path = path.join(REGRESSION_PATH, "hooks.json")
    # This should result in the same signal in test_hooks
    fx_regression(augmented_signal.audio_data, reg_path, check_against_regression_data)

    with pytest.raises(AudioSignalException):
        signal.make_effect("fail")

def test_filter_function_pass():
    # this test is for 100% coverage
    function = effects.FilterFunction()
    dummy_item = None
    function.func(dummy_item)
