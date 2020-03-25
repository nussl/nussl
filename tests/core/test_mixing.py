import nussl
import numpy as np
import pytest


def test_pan_audio_signal(mix_and_sources):
    mix, sources = mix_and_sources
    sources = list(sources.values())

    panned_audio = nussl.mixing.pan_audio_signal(sources[0], -45)

    zeros = np.zeros_like(panned_audio.audio_data[0])
    sum_ch = np.sum(panned_audio.audio_data, axis=0)
    assert np.allclose(panned_audio.audio_data[1], zeros)
    assert np.allclose(panned_audio.audio_data[0], sum_ch)

    panned_audio = nussl.mixing.pan_audio_signal(sources[0], 45)

    zeros = np.zeros_like(panned_audio.audio_data[0])
    sum_ch = np.sum(panned_audio.audio_data, axis=0)
    assert np.allclose(panned_audio.audio_data[0], zeros)
    assert np.allclose(panned_audio.audio_data[1], sum_ch)

    pytest.raises(ValueError, nussl.mixing.pan_audio_signal, mix, -46)
    pytest.raises(ValueError, nussl.mixing.pan_audio_signal, mix, 46)


def test_delay_audio_signal(mix_and_sources):
    mix, sources = mix_and_sources
    sources = list(sources.values())

    a = nussl.mixing.pan_audio_signal(sources[0], -35)
    b = nussl.mixing.pan_audio_signal(sources[1], 15)
    mix = a + b

    delays = [np.random.randint(1, 1000) for _ in range(mix.num_channels)]

    delayed_audio = nussl.mixing.delay_audio_signal(mix, delays)

    for i, d in enumerate(delays):
        _est = delayed_audio.audio_data[i]
        _true = mix.audio_data[i, :-d]

        assert np.allclose(_est[d:], _true)

    pytest.raises(ValueError, nussl.mixing.delay_audio_signal, mix, [0, 0, 0])
    pytest.raises(ValueError, nussl.mixing.delay_audio_signal, mix, [0, -10, 0])
    pytest.raises(ValueError, nussl.mixing.delay_audio_signal, mix, [0, .1, 2.0])
