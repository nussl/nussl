import nussl
import scipy.io.wavfile as wav
import pytest
import numpy as np
import tempfile
import librosa
from nussl.core.audio_signal import AudioSignalException
import copy

sr = nussl.constants.DEFAULT_SAMPLE_RATE
dur = 3  # seconds
length = dur * sr


def test_load(benchmark_audio):
    # Load from file
    a = nussl.AudioSignal(benchmark_audio['K0140.wav'])
    b = nussl.AudioSignal()
    b.load_audio_from_file(benchmark_audio['K0140.wav'])

    assert (np.array_equal(a.audio_data, b.audio_data))
    assert (a.sample_rate == b.sample_rate)

    # Load from array
    ref_sr, ref_data = wav.read(benchmark_audio['K0140.wav'])
    c = nussl.AudioSignal(audio_data_array=ref_data, sample_rate=ref_sr)

    pytest.raises(
        Exception, nussl.AudioSignal, benchmark_audio['K0140.wav'], ref_data)

    pytest.raises(
        Exception, nussl.AudioSignal,
        path_to_input_file=benchmark_audio['K0140.wav'], audio_data_array=ref_data)

    d = nussl.AudioSignal()
    d.load_audio_from_array(ref_data, ref_sr)

    assert (np.array_equal(c.audio_data, d.audio_data))
    assert (c.sample_rate == d.sample_rate)
    assert (b.sample_rate == c.sample_rate)
    assert (np.array_equal(b.audio_data, c.audio_data))


def test_audio_data_setters(benchmark_audio):
    # Load from file
    a = nussl.AudioSignal(benchmark_audio['K0140.wav'])

    def dummy_a(signal):
        signal.audio_data = np.ones((10, 10, 10))

    pytest.raises(AudioSignalException, dummy_a, a)

    def dummy_b(signal):
        signal.audio_data = [1, 2, 3]

    pytest.raises(AudioSignalException, dummy_b, a)

    def dummy_c(signal):
        _audio_data = np.ones((10, 10, 10))
        _audio_data[0] = np.inf
        signal.audio_data = _audio_data

    pytest.raises(AudioSignalException, dummy_c, a)


def test_load_audio_from_file(benchmark_audio):
    # Do some preliminary checks
    signal_info = {}
    for key, path in benchmark_audio.items():
        ref_sr, ref_data = wav.read(path)
        ref_dur = len(ref_data) / ref_sr
        n_chan = 1 if len(ref_data.shape) == 1 else ref_data.shape[1]

        signal_info[key] = {
            'duration': ref_dur,
            'sample_rate': ref_sr,
            'length': len(ref_data),
            'n_chan': n_chan
        }

        a = nussl.AudioSignal()
        a.load_audio_from_file(path)

        assert a.signal_length == len(ref_data)
        assert a.num_channels == n_chan
        assert a.sample_rate == ref_sr
        assert np.isclose(a.signal_duration, ref_dur)
        assert a.active_region_is_default

    # Test different offsets
    percentages = [0.1, 0.25, 0.4, 0.5, 0.75, 0.9]
    for key, path in benchmark_audio.items():
        for start in percentages:
            offset = start * signal_info[key]['duration']
            ref_length = int(round(signal_info[key]['length'] - offset *
                                   signal_info[key]['sample_rate']))

            a = nussl.AudioSignal()
            a.load_audio_from_file(path, offset=offset)

            # Sometimes ref_length is off by 1 due to rounding
            assert abs(a.signal_length - ref_length) <= 1

    # Test different durations
    for key, path in benchmark_audio.items():
        for start in percentages:
            duration = start * signal_info[key]['duration']
            ref_length = int(round(duration * signal_info[key]['sample_rate']))

            a = nussl.AudioSignal()
            a.load_audio_from_file(path, duration=duration)

            # Sometimes ref_length is off by 1 due to rounding
            assert abs(a.signal_length - ref_length) <= 1

    # Test offsets and durations
    percentages = np.arange(0.01, 0.51, 0.05)
    for key, path in benchmark_audio.items():
        for start in percentages:
            for duration in percentages:
                offset = start * signal_info[key]['duration']
                duration = duration * signal_info[key]['duration']
                ref_length = int(round(duration * signal_info[key]['sample_rate']))

                a = nussl.AudioSignal()
                a.load_audio_from_file(path, offset=offset, duration=duration)

                # Sometimes ref_length is off by 2 due to rounding
                assert abs(a.signal_length - ref_length) <= 2

    # Test error cases
    path = benchmark_audio['K0140.wav']
    sr, data = wav.read(path)
    dur = len(data) / sr
    offset = dur + 1.0
    a = nussl.AudioSignal()

    pytest.raises(Exception, a.load_audio_from_file, path, offset=offset)
    pytest.raises(Exception, a.load_audio_from_file, 'fake_path')

    # Make sure this is okay
    offset = dur / 2.0
    duration = dur
    a = nussl.AudioSignal()
    a.load_audio_from_file(path, offset=offset, duration=duration)


def test_write_to_file(benchmark_audio):
    for key, path in benchmark_audio.items():
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as f:
            dummy = nussl.AudioSignal()
            pytest.raises(
                AudioSignalException, dummy.write_audio_to_file, f.name)

            a = nussl.AudioSignal(path)
            a.write_audio_to_file(f.name)
            b = nussl.AudioSignal(f.name)

            assert (a.sample_rate == b.sample_rate)


def test_write_array_to_file(benchmark_audio):
    for key, path in benchmark_audio.items():
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as f:
            sr, data = wav.read(path)
            a = nussl.AudioSignal(audio_data_array=data, sample_rate=sr)
            a.write_audio_to_file(f.name)
            b = nussl.AudioSignal(f.name)

            assert (a.sample_rate == b.sample_rate)
            assert (np.allclose(a.audio_data, b.audio_data))


def test_write_sample_rate(benchmark_audio):
    for key, path in benchmark_audio.items():
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as f:
            a = nussl.AudioSignal(path)
            sample_rate = a.sample_rate // 2
            a.write_audio_to_file(f.name, sample_rate=sample_rate)
            b = nussl.AudioSignal(f.name)

            assert (b.sample_rate == sample_rate)


def test_resample(benchmark_audio):
    # Check that sample rate property changes
    for key, path in benchmark_audio.items():
        a = nussl.AudioSignal(path)
        b = nussl.AudioSignal(path)
        b.resample(a.sample_rate / 2)
        assert (b.sample_rate == a.sample_rate / 2)

        pytest.warns(UserWarning, a.resample, a.sample_rate)


def test_resample_on_load_from_file(benchmark_audio):
    # Test resample right when loading from file vs resampling after loading
    path = [benchmark_audio[key] for key in benchmark_audio][0]
    a = nussl.AudioSignal(path)
    a.resample(48000)
    b = nussl.AudioSignal()
    b.load_audio_from_file(path, new_sample_rate=48000)
    assert (a.sample_rate == b.sample_rate)
    assert (np.allclose(a.audio_data, b.audio_data))

    pytest.warns(UserWarning, nussl.AudioSignal, path, sample_rate=16000)


def test_resample_vs_librosa_load(benchmark_audio):
    # Check against librosa load function
    path = [benchmark_audio[key] for key in benchmark_audio][0]
    a = nussl.AudioSignal(path)
    a.resample(48000)
    b_audio_data, b_sample_rate = librosa.load(path, sr=48000)
    assert (a.sample_rate == b_sample_rate)
    assert (np.allclose(a.audio_data, b_audio_data))


def test_default_sr_on_load_from_array(benchmark_audio):
    # Check that the default sample rate is set when no sample rate is provided load_audio_from_array
    path = [benchmark_audio[key] for key in benchmark_audio][0]
    sr, data = wav.read(path)
    a = nussl.AudioSignal()
    a.load_audio_from_array(data)
    assert (a.sample_rate == nussl.constants.DEFAULT_SAMPLE_RATE)


def test_str(benchmark_audio):
    for key, path in benchmark_audio.items():
        a = nussl.AudioSignal(path)
        b = nussl.AudioSignal(path)
        assert (str(a) == str(b))


def test_rms():
    ans = np.sqrt(2.0) / 2.0

    num_samples = nussl.constants.DEFAULT_SAMPLE_RATE  # 1 second
    np_sin = np.sin(np.linspace(0, 100 * 2 * np.pi, num_samples))  # Freq = 100 Hz
    signal = np.vstack([np_sin, -np_sin])

    sig = nussl.AudioSignal(audio_data_array=signal)
    assert np.allclose([ans, ans], sig.rms(), atol=1e-06)

    n_seconds = 5
    dur = num_samples * n_seconds  # 5 sec
    np_sin = np.sin(np.linspace(0, 100 * 2 * np.pi, dur))  # Freq = 100 Hz

    sig = nussl.AudioSignal(audio_data_array=np_sin)
    rms = sig.rms(win_len=num_samples)
    answer = np.array([ans for _ in range(n_seconds*2+1)])
    assert np.allclose(answer, rms, atol=1e-06)

def test_loudness(benchmark_audio):
    for key, path in benchmark_audio.items():
        a = nussl.AudioSignal(path)
        a.loudness()

def test_peak_normalize():
    num_samples = nussl.constants.DEFAULT_SAMPLE_RATE  # 1 second
    np_sin = np.sin(np.linspace(0, 100 * 2 * np.pi, num_samples))  # Freq = 100 Hz
    signal = np.vstack([np_sin, -np_sin])

    sig = nussl.AudioSignal(audio_data_array=signal)
    sig.peak_normalize()

    assert sig.audio_data.max() == 1.0

def test_to_mono():
    """
    Test functionality and correctness of AudioSignal.to_mono() function.
    Returns:

    """
    num_samples = nussl.constants.DEFAULT_SAMPLE_RATE  # 1 second
    sin1 = np.sin(np.linspace(0, 100 * 2 * np.pi, num_samples))  # Freq = 100 Hz

    sig1 = nussl.AudioSignal(audio_data_array=sin1)
    assert (sig1.num_channels == 1)
    sig1.to_mono(overwrite=True)
    assert (sig1.num_channels == 1)

    sin2 = -1 * sin1

    sines = np.vstack((sin1, sin2))
    sig2 = nussl.AudioSignal(audio_data_array=sines)
    assert (sig2.num_channels == 2)
    new_sig = sig2.to_mono(overwrite=False)
    assert (new_sig.num_channels == 1)
    assert (sig2.num_channels == 2)
    sig2.to_mono()
    assert (sig2.num_channels == 1)

    assert (np.allclose([0.0] * len(sig2), sig2.audio_data))


def test_to_mono_channel_dimension(benchmark_audio):
    """
    Test functionality and correctness of AudioSignal.to_mono() function.
    Returns:

    """
    path = benchmark_audio['dev1_female3_inst_mix.wav']
    # Load input file
    signal = nussl.AudioSignal(path)
    signal.stft_params = signal.stft_params
    signal_stft = signal.stft()
    assert (signal_stft.shape[nussl.constants.STFT_CHAN_INDEX] == 2)

    signal.to_mono(overwrite=True)
    signal.stft_params = signal.stft_params
    signal_stft = signal.stft()
    assert (signal_stft.shape[nussl.constants.STFT_CHAN_INDEX] == 1)


def test_get_channel():
    # Here we're setting up signals with 1 to 8 channels
    # Each channel has a sine wave of different frequency in it

    # This is the frequencies for our different channels
    max_n_channels = 8
    freq_multiple = 300
    freqs = [i * freq_multiple for i in range(max_n_channels)]

    # Make the signals and test
    for f in range(1, max_n_channels):  # 1-8 channel mixtures
        sig = np.array(
            [np.sin(np.linspace(0, i * 2 * np.pi, length))
             for i in freqs[:f]
             ])
        _get_channel_helper(sig, len(sig))


def _get_channel_helper(signal, n_channels):
    a = nussl.AudioSignal(audio_data_array=signal)

    # Check that we are counting our channels correctly
    assert a.num_channels == n_channels

    # Check that we can get every channel with AudioSignal.get_channel()
    for i, ch in enumerate(signal):
        assert np.array_equal(a.get_channel(i), ch)

    # Check that attempting to get higher channels raises exception
    for i in range(n_channels, n_channels + 10):
        pytest.raises(AudioSignalException,
                      a.get_channel, i)

    # Check that attempting to get lower channels raises exception
    for i in range(-1, -11, -1):
        pytest.raises(AudioSignalException,
                      a.get_channel, i)

    # Check that AudioSignal.get_channels() generator works
    i = 0
    for ch in a.get_channels():
        assert np.array_equal(ch, signal[i, :])
        new_signal = a.make_audio_signal_from_channel(i)
        assert np.array_equal(ch, new_signal.audio_data[0])
        i += 1
    assert i == a.num_channels


def test_active_region(benchmark_audio):
    a = nussl.AudioSignal(benchmark_audio['K0140.wav'])
    original_length = a.signal_length
    a.set_active_region(
        int(.1 * a.sample_rate), int(.5 * a.sample_rate))

    assert (
            a.signal_length ==
            (int(.5 * a.sample_rate) - int(.1 * a.sample_rate)))
    assert not a.active_region_is_default

    a.set_active_region_to_default()
    assert a.signal_length == original_length
    assert a.active_region_is_default


def test_audio_signal_copy(benchmark_audio):
    for key, path in benchmark_audio.items():
        signal = nussl.AudioSignal(path)

        new_signal = signal.make_copy_with_audio_data(signal.audio_data)
        assert np.allclose(
            new_signal.audio_data, signal.audio_data)

        signal.set_active_region(0, 1000)
        pytest.warns(UserWarning, signal.make_copy_with_audio_data,
                     signal.audio_data)

        signal.set_active_region_to_default()
        audio_data = signal.audio_data[0]
        pytest.warns(UserWarning, signal.make_copy_with_audio_data,
                     audio_data)


def test_audio_signal_utilities(benchmark_audio):
    a1 = nussl.AudioSignal(benchmark_audio['K0140.wav'])
    a2 = nussl.AudioSignal(benchmark_audio['K0140.wav'])
    b = nussl.AudioSignal(benchmark_audio['K0149.wav'])

    def check_if_exc(signal, func, *args):
        signal.set_active_region(0, 100)
        pytest.raises(AudioSignalException, func, *args)
        signal.set_active_region_to_default()

    a1.concat(b)
    assert (a1.signal_length == (a2.signal_length + b.signal_length))

    truncate_to = a1.signal_length // 2
    check_if_exc(a1, a1.truncate_samples, truncate_to)
    a1.truncate_samples(truncate_to)
    assert (a1.signal_length == truncate_to)

    a1.truncate_samples(10 * truncate_to)
    assert (a1.signal_length == truncate_to)

    truncate_to_sec = a1.signal_duration / 2
    check_if_exc(a1, a1.truncate_seconds, truncate_to_sec)
    a1.truncate_seconds(truncate_to_sec)
    assert (a1.signal_duration == truncate_to_sec)

    before, after = 1000, 1000
    check_if_exc(a2, a2.crop_signal, before, after)
    old_duration = a2.signal_length
    a2.crop_signal(before, after)
    assert (a2.signal_length == old_duration - 2000)

    before, after = 1000, 1000
    check_if_exc(a2, a2.zero_pad, before, after)
    old_duration = a2.signal_length
    a2.zero_pad(before, after)
    assert (a2.signal_length == old_duration + 2000)


def test_arithmetic(benchmark_audio):
    # These signals have two different lengths
    a = nussl.AudioSignal(benchmark_audio['K0140.wav'])
    b = nussl.AudioSignal(benchmark_audio['K0149.wav'])

    pytest.raises(AudioSignalException, a.add, b)
    pytest.raises(AudioSignalException, a.subtract, b)
    pytest.raises(AudioSignalException, lambda a, b: a + b, a, b)
    pytest.raises(AudioSignalException, lambda a, b: a - b, a, b)
    pytest.raises(AudioSignalException, lambda a, b: a * b, a, b)
    pytest.raises(AudioSignalException, lambda a, b: a / b, a, b)
    pytest.raises(AudioSignalException, lambda a, b: a / a, a, a)

    assert (np.allclose((a + a).audio_data, a.audio_data + a.audio_data))
    assert (np.allclose((a - a).audio_data, a.audio_data - a.audio_data))

    c = a * 2
    assert (np.allclose(c.audio_data, a.audio_data * 2))

    d = copy.copy(a)
    d *= 2
    assert (np.allclose(c.audio_data, d.audio_data))

    c = a / 2
    assert (np.allclose(c.audio_data, a.audio_data / 2))
    d = copy.copy(a)
    d /= 2
    assert (np.allclose(c.audio_data, d.audio_data))

    a2 = nussl.AudioSignal(benchmark_audio['K0140.wav'])
    b2 = nussl.AudioSignal(benchmark_audio['K0149.wav'])
    a3 = nussl.AudioSignal(benchmark_audio['K0140.wav'])
    a3.audio_data *= .5

    assert (a == a2)
    assert (a2 != b2)
    assert (a2 != a3)

    assert len(a2) == a2.audio_data.shape[-1]

    a2 -= a3
    assert (a2 == a3)

    a2 += a3
    assert (a2 == a)

    a2.set_active_region(0, 100)
    pytest.raises(AudioSignalException, lambda a, b: a + b, a, a2)

    a2.set_active_region_to_default()
    a2.resample(a2.sample_rate // 2)
    pytest.raises(AudioSignalException, lambda a, b: a + b, a, a2)

    c1 = nussl.AudioSignal(benchmark_audio['dev1_female3_inst_mix.wav'])
    assert c1.is_stereo
    c1.to_mono()
    c2 = nussl.AudioSignal(benchmark_audio['dev1_female3_inst_mix.wav'])
    pytest.raises(AudioSignalException, lambda a, b: a + b, c1, c2)

    res1 = a2 + 0
    res2 = 0 + a2

    assert (res1 == res2)

    def dummy(a, b):
        a *= b

    pytest.raises(AudioSignalException, dummy, a, [0, 2])


def test_properties(benchmark_audio):
    a = nussl.AudioSignal()
    assert a.signal_duration is None
    assert a.signal_length is None
    assert a._signal_length is None
    assert a.num_channels is None
    assert a.time_vector is None
    assert a.file_name is None
    assert not a.has_data

    a = nussl.AudioSignal(benchmark_audio['K0140.wav'])
    assert a.has_data
    assert a.file_name == 'K0140.wav'
    assert len(a.time_vector) == a.audio_data.shape[-1]

    pytest.raises(AudioSignalException, lambda x: x.stft_length, a)
    pytest.raises(AudioSignalException, lambda x: x.time_bins_vector, a)
    pytest.raises(AudioSignalException, lambda x: x.freq_vector, a)
    a.stft()

    assert a.stft_length == a.stft_data.shape[1]
    assert len(a.time_bins_vector) == a.stft_length
    assert len(a.freq_vector) == a.stft_data.shape[0]

    assert len(a.freq_vector) == a.stft_data.shape[0]

    a.audio_data = None
    assert a.has_data
