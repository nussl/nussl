import nussl
import numpy as np
import pytest

def test_convolve(mix_and_sources, check_against_regression_data):
    signal, _ = mix_and_sources

    def _impulse(data):
        data[:, 0] = 1
        impulse = nussl.AudioSignal(
            audio_data_array=data, 
            sample_rate=signal.sample_rate
        )
        return impulse

    # Identity
    output = signal.convolve(_impulse(np.zeros((1, 100))))
    assert output == signal

    # Channel mismatch should raise errors
    signal.audio_data = np.vstack([signal.audio_data, signal.audio_data])
    pytest.raises(RuntimeError, signal.convolve, _impulse(np.zeros((3, 100))))

    # Multichannel identity
    output = signal.convolve(_impulse(np.zeros((2, 100))))
    assert output == signal

def test_pan_audio_signal(mix_and_sources):
    mix, sources = mix_and_sources
    sources = list(sources.values())

    panned_audio = nussl.mixing.pan_audio_signal(sources[0], -45)

    zeros = np.zeros_like(panned_audio.audio_data[0])
    sum_ch = np.sum(panned_audio.audio_data, axis=0)
    assert np.allclose(panned_audio.audio_data[1], zeros)
    assert np.allclose(panned_audio.audio_data[0], sum_ch)

    hook_panned_audio = sources[0].pan(-45)
    assert hook_panned_audio == panned_audio

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

    hook_delayed_audio = mix.delay(delays)
    assert hook_delayed_audio == delayed_audio

    pytest.raises(ValueError, nussl.mixing.delay_audio_signal, mix, [0, 0, 0])
    pytest.raises(ValueError, nussl.mixing.delay_audio_signal, mix, [0, -10, 0])
    pytest.raises(ValueError, nussl.mixing.delay_audio_signal, mix, [0, .1, 2.0])

def test_mix_audio_signals(mix_and_sources):
    mix, sources = mix_and_sources
    sources = list(sources.values())
    fg = sources[0]
    bg = sources[1]

    mix, fg_after, bg_after = fg.mix(bg, snr=10)
    assert np.allclose(fg_after.loudness() - bg_after.loudness(), 10)

    # Try negative SNR
    mix, fg_after, bg_after = fg.mix(bg, snr=-10)
    assert np.allclose(fg_after.loudness() - bg_after.loudness(), -10)

    # Warns on big pos/neg SNR
    pytest.warns(UserWarning, fg.mix, bg, snr=1000)
    pytest.warns(UserWarning, fg.mix, bg, snr=-1000)    
