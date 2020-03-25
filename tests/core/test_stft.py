import nussl
import scipy.io.wavfile as wav
import pytest
import numpy as np
import tempfile
import librosa
from nussl.core.audio_signal import AudioSignalException, STFTParams
from nussl.core.constants import ALL_WINDOWS
from nussl import AudioSignal
from scipy.signal import check_COLA
import copy
import itertools

sr = nussl.constants.DEFAULT_SAMPLE_RATE
dur = 3  # seconds
length = dur * sr
stft_tol = 1e-6
n_ch = 2

freq = 30
sine_wave = np.sin(np.linspace(0, freq * 2 * np.pi, length))

# Define my window lengths to be powers of 2, ranging from 128 to 8192 samples
win_min = 7  # 2 ** 7  =  128
win_max = 11  # 2 ** 13 = 4096
win_lengths = [2 ** i for i in range(win_min, win_max + 1)]

win_length_32ms = int(2 ** (np.ceil(np.log2(nussl.constants.DEFAULT_WIN_LEN_PARAM * sr))))
win_lengths.append(win_length_32ms)

# pick hop lengths in terms of window length. 1 = one full window. .1 = 1/10th of a window
hop_length_ratios = [0.75, 0.5, 0.3, 0.25, 0.1]

window_types = ALL_WINDOWS

signals = []

combos = itertools.product(win_lengths, hop_length_ratios, window_types)


@pytest.fixture
def signals(benchmark_audio):
    signals_ = []
    # noisy signal
    noise = (np.random.rand(n_ch, length) * 2) - 1
    signals_.append(noise)

    # ones signal
    ones = np.ones(length)
    signals_.append(ones)

    # audio files
    for key, path in benchmark_audio.items():
        _s = nussl.AudioSignal(path, duration=dur)
        signals_.append(_s.audio_data)

    yield signals_


@pytest.mark.parametrize("combo", combos)
def test_stft_istft_combo(combo, signals):
    win_length = combo[0]
    hop_length = int(combo[0] * combo[1])
    win_type = combo[2]
    window = nussl.AudioSignal.get_window(combo[2], win_length)

    if not check_COLA(window, win_length, win_length - hop_length):
        assert True

    for signal in signals:
        _check_stft_istft_allclose(
            signal, win_length, hop_length, win_type
        )


def test_stft_copy(signals):
    for audio_data in signals:
        signal = AudioSignal(
            audio_data_array=audio_data, sample_rate=sr)
        stft = signal.stft()

        new_signal = signal.make_copy_with_stft_data(stft)
        new_signal.istft(truncate_to_length=signal.signal_length)
        assert np.allclose(
            new_signal.audio_data, signal.audio_data, atol=stft_tol)

        signal.set_active_region(0, 1000)
        pytest.warns(UserWarning, signal.make_copy_with_stft_data,
                     stft)

        signal.set_active_region_to_default()
        stft = stft[0]
        pytest.warns(UserWarning, signal.make_copy_with_stft_data,
                     stft)

        def dummy_a(signal):
            signal.stft_data = np.abs(signal.stft_data)

        pytest.warns(UserWarning, dummy_a, signal)

        def dummy_b(signal):
            signal.stft_data = np.ones((2, 100, 100, 2))

        pytest.raises(AudioSignalException, dummy_b, signal)

        def dummy_c(signal):
            signal.stft_data = np.ones((2,))

        pytest.raises(AudioSignalException, dummy_c, signal)

        def dummy_d(signal):
            signal.stft_data = [1, 2, 3, 4]

        pytest.raises(AudioSignalException, dummy_d, signal)


def test_stft_features(signals):
    for audio_data in signals:
        signal = AudioSignal(
            audio_data_array=audio_data, sample_rate=sr)
        pytest.raises(AudioSignalException, signal.get_stft_channel, 0)
        pytest.raises(AudioSignalException, signal.ipd_ild_features)
        pytest.raises(AudioSignalException,
                      lambda x: x.log_magnitude_spectrogram_data,
                      signal)
        pytest.raises(AudioSignalException,
                      lambda x: x.magnitude_spectrogram_data,
                      signal)
        pytest.raises(AudioSignalException,
                      lambda x: x.power_spectrogram_data,
                      signal)

        signal.stft()
        ref_mag_spec = np.abs(signal.stft_data)
        tst_mag_spec = signal.magnitude_spectrogram_data
        assert np.allclose(ref_mag_spec, tst_mag_spec)

        ref_mag_spec = np.abs(signal.stft_data)
        tst_mag_spec = signal.magnitude_spectrogram_data
        assert np.allclose(ref_mag_spec, tst_mag_spec)

        ref_log_spec = 20 * np.log10(ref_mag_spec + 1e-8)
        tst_log_spec = signal.log_magnitude_spectrogram_data
        assert np.allclose(ref_log_spec, tst_log_spec)

        ref_pow_spec = ref_mag_spec ** 2
        tst_pow_spec = signal.power_spectrogram_data
        assert np.allclose(ref_pow_spec, tst_pow_spec)

        for ch in range(signal.num_channels):
            tst_mag_spec = signal.get_magnitude_spectrogram_channel(ch)
            tst_pow_spec = signal.get_power_spectrogram_channel(ch)

            assert np.allclose(ref_mag_spec[..., ch], tst_mag_spec)
            assert np.allclose(ref_pow_spec[..., ch], tst_pow_spec)

        if signal.is_mono:
            pytest.raises(AudioSignalException, signal.ipd_ild_features)
        else:
            _, _ = signal.ipd_ild_features(0, 1)


def test_stft_istft_defaults(benchmark_audio, atol=stft_tol):
    dummy = nussl.AudioSignal()
    pytest.raises(AudioSignalException, dummy.stft)

    a = nussl.AudioSignal(audio_data_array=sine_wave)
    pytest.raises(AudioSignalException, a.istft)
    a.stft()
    a.istft()

    a = nussl.AudioSignal(audio_data_array=sine_wave)
    a.stft()
    calc_sine = a.istft(overwrite=False)

    assert np.allclose(a.audio_data, calc_sine, atol=stft_tol)

    # also load another object with stft_data
    b = nussl.AudioSignal(stft=a.stft(), sample_rate=a.sample_rate)
    b.istft()
    min_length = min(b.audio_data.shape[1], a.audio_data.shape[1])

    assert np.allclose(
        a.audio_data[:, :min_length],
        b.audio_data[:, :min_length],
        atol=stft_tol)

    for key, path in benchmark_audio.items():
        a = nussl.AudioSignal(path)
        a.stft()
        recon = a.istft(overwrite=False)
        assert np.allclose(a.audio_data, recon, atol=stft_tol)


def test_stft_params_setter():
    dummy = nussl.AudioSignal(sample_rate=44100)
    dummy.stft_params = None

    default_win_len = int(
        2 ** (np.ceil(np.log2(
            nussl.constants.DEFAULT_WIN_LEN_PARAM * dummy.sample_rate))
        ))
    default_hop_len = default_win_len // 4
    default_win_type = nussl.constants.WINDOW_DEFAULT

    default_stft_params = STFTParams(
        window_length=default_win_len,
        hop_length=default_hop_len,
        window_type=default_win_type
    )

    assert dummy.stft_params == default_stft_params

    dummy.stft_params = STFTParams(
        window_length=4096,
    )
    assert dummy.stft_params.window_length == 4096
    assert dummy.stft_params.hop_length == default_hop_len
    assert dummy.stft_params.window_type == default_win_type

    dummy.stft_params = STFTParams(
        window_length=4096,
        hop_length=1024
    )
    assert dummy.stft_params.window_length == 4096
    assert dummy.stft_params.hop_length == 1024
    assert dummy.stft_params.window_type == default_win_type

    dummy.stft_params = STFTParams(
        window_length=4096,
        hop_length=1024,
        window_type='triang'
    )
    assert dummy.stft_params.window_length == 4096
    assert dummy.stft_params.hop_length == 1024
    assert dummy.stft_params.window_type == 'triang'

    def dummy_set(x):
        x.stft_params = ['test123']

    pytest.raises(ValueError, dummy_set, dummy)


def _check_stft_istft_allclose(audio_data, win_length, hop_length, win_type):
    stft_params = STFTParams(
        window_length=win_length, hop_length=hop_length, window_type=win_type
    )
    signal = AudioSignal(
        audio_data_array=audio_data, sample_rate=sr, stft_params=stft_params)
    signal.stft()
    recon = signal.istft(overwrite=False)

    assert np.allclose(signal.audio_data, recon, atol=stft_tol)
