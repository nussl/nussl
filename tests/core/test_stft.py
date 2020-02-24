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

sr = nussl.DEFAULT_SAMPLE_RATE
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

# figure out what the length a 40 millisecond window would be (in samples). We pick 40 ms because...
# it is 1/25th of a second. This gives a window length that can hold 1 full cycle of a 25 hz signal
# Of course, human hearing extends down to 20 Hz....but at least this is lower than the lowest note
# on a piano (27.5 Hz for equal temperament A440 tuning)
win_length_40ms = int(2 ** (np.ceil(np.log2(nussl.DEFAULT_WIN_LEN_PARAM * sr))))
win_lengths.append(win_length_40ms)

# pick hop lengths in terms of window length. 1 = one full window. .1 = 1/10th of a window
hop_length_ratios = [0.75, 0.5, 0.3, 0.25, 0.1]

window_types = ALL_WINDOWS

signals = []

combos = itertools.product(win_lengths, hop_length_ratios, window_types)

@pytest.fixture
def signals(benchmark_audio):
    signals = []
    # noisy signal
    noise = (np.random.rand(n_ch, length) * 2) - 1
    noise = AudioSignal(audio_data_array=noise, sample_rate=sr)
    signals.append(noise)

    # ones signal
    ones = np.ones(length)
    ones = AudioSignal(audio_data_array=ones, sample_rate=sr)
    signals.append(ones)

    # audio files
    for key, path in benchmark_audio.items():
        _s = nussl.AudioSignal(path, duration=dur)
        signals.append(_s)

    yield signals

@pytest.mark.parametrize("combo", combos)
def test_stft_istft_combo(combo, signals):
    win_length = combo[0]
    hop_length = int(combo[0] * combo[1])
    win_type = combo[2]

    if not check_COLA(win_type, win_length, win_length - hop_length):
        assert True

    for signal in signals:
        _check_stft_istft_allclose(
            signal, win_length, hop_length, win_type
        )


def test_stft_istft_defaults(benchmark_audio, atol=stft_tol):
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

def _check_stft_istft_allclose(signal, win_length, hop_length, win_type):
    stft_params = STFTParams(
        window_length=win_length, hop_length=hop_length, window_type=win_type
    )
    signal.stft_params = stft_params
    signal.stft()
    recon = signal.istft(overwrite=False)
    
    assert np.allclose(signal.audio_data, recon, atol=stft_tol)
