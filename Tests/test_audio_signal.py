import unittest
import nussl
import numpy as np
import scipy.io.wavfile as wav


class TestAudioSignal(unittest.TestCase):
    sr = nussl.constants.DEFAULT_SAMPLE_RATE
    dur = 3  # seconds
    length = dur * sr

    def setUp(self):
        self.path = "../Input/k0140_int.wav"
        self.out_path = '../Output/test_write.wav'

    def test_load(self):
        # Load from file
        a = nussl.AudioSignal(self.path)
        b = nussl.AudioSignal()
        b.load_audio_from_file(self.path)

        assert (np.array_equal(a.audio_data, b.audio_data))
        assert (a.sample_rate == b.sample_rate)

        # Load from array
        sr, data = wav.read(self.path)
        c = nussl.AudioSignal(audio_data_array=data, sample_rate=sr)
        d = nussl.AudioSignal()
        d.load_audio_from_array(data, sr)

        assert (np.array_equal(c.audio_data, d.audio_data))
        assert (c.sample_rate == d.sample_rate)
        assert (b.sample_rate == c.sample_rate)
        assert (np.array_equal(b.audio_data, c.audio_data))

    def test_write_to_file_path1(self):
        a = nussl.AudioSignal(self.path)
        a.write_audio_to_file(self.out_path)
        b = nussl.AudioSignal(self.out_path)

        assert (a.sample_rate == b.sample_rate)
        assert (np.allclose(a.audio_data, b.audio_data))

    def test_write_to_file_path2(self):
        a = nussl.AudioSignal()
        a.load_audio_from_file(self.path)
        a.write_audio_to_file(self.out_path)
        b = nussl.AudioSignal(self.out_path)

        assert (a.sample_rate == b.sample_rate)
        assert (np.allclose(a.audio_data, b.audio_data))

    def test_write_to_file_array1(self):
        sr, data = wav.read(self.path)
        a = nussl.AudioSignal(audio_data_array=data, sample_rate=sr)
        a.write_audio_to_file(self.out_path)
        b = nussl.AudioSignal(self.out_path)

        assert (a.sample_rate == b.sample_rate)
        assert (np.allclose(a.audio_data, b.audio_data))

    def test_write_to_file_array2(self):
        sr, data = wav.read(self.path)
        a = nussl.AudioSignal()
        a.load_audio_from_array(data, sr)
        a.write_audio_to_file(self.out_path)
        b = nussl.AudioSignal(self.out_path)

        assert (a.sample_rate == b.sample_rate)
        assert (np.allclose(a.audio_data, b.audio_data))

    def test_write_sample_rate(self):
        a = nussl.AudioSignal(self.path)
        sample_rate = a.sample_rate / 2
        a.write_audio_to_file(self.out_path, sample_rate=sample_rate)
        b = nussl.AudioSignal(self.out_path)

        assert (b.sample_rate == sample_rate)

    freq = 30
    sine_wave = np.sin(np.linspace(0, freq * 2 * np.pi, length))

    def test_stft_istft_simple1(self):
        a = nussl.AudioSignal(audio_data_array=self.sine_wave)
        a.stft()
        a.istft()

    def test_stft_istft_simple2(self):
        a = nussl.AudioSignal(audio_data_array=self.sine_wave)

        # complete OLA
        a.stft_params.window_type = nussl.constants.WINDOW_RECTANGULAR
        a.stft_params.hop_length = a.stft_params.window_length

        a.stft()
        calc_sine = a.istft(overwrite=False)

        l = a.stft_params.hop_length // 2
        # diff = a.audio_data[0, 0:a.signal_length] - calc_sine[0, l:a.signal_length+l]

        assert (np.allclose(a.audio_data[0, 0:a.signal_length], calc_sine[0, l:a.signal_length + l]))

    def test_rms(self):
        ans = np.sqrt(2.0) / 2.0

        num_samples = nussl.DEFAULT_SAMPLE_RATE  # 1 second
        np_sin = np.sin(np.linspace(0, 100 * 2 * np.pi, num_samples))  # Freq = 100 Hz

        sig = nussl.AudioSignal(audio_data_array=np_sin)
        assert (np.isclose(ans, sig.rms(), atol=1e-06))

    def test_to_mono(self):
        """
        Test functionality and correctness of AudioSignal.to_mono() function.
        Returns:

        """
        num_samples = nussl.DEFAULT_SAMPLE_RATE  # 1 second
        sin1 = np.sin(np.linspace(0, 100 * 2 * np.pi, num_samples))  # Freq = 100 Hz

        sig1 = nussl.AudioSignal(audio_data_array=sin1)
        assert (sig1.num_channels == 1)
        sig1.to_mono(overwrite=True)
        assert (sig1.num_channels == 1)

        sin2 = -1 * sin1

        sines = np.vstack((sin1, sin2))
        sig2 = nussl.AudioSignal(audio_data_array=sines)
        assert (sig2.num_channels == 2)
        sig2.to_mono()
        assert (sig2.num_channels == 2)
        sig2.to_mono(overwrite=False)
        assert (sig2.num_channels == 2)
        sig2.to_mono(overwrite=True)
        assert (sig2.num_channels == 1)
        assert (np.allclose([0.0] * len(sig2), sig2.audio_data))

    def test_stft(self):
        """
        Test some basic functionality of the STFT interface for the AudioSignal object
        All verification of the STFT calculation is done in test_spectral_utils file.
        """
        signal = nussl.AudioSignal(audio_data_array=self.sine_wave)

        self.assertTrue(signal.stft_data is None)
        signal.stft(overwrite=False)
        self.assertTrue(signal.stft_data is None)
        stft = signal.stft()
        self.assertTrue(signal.stft_data is not None)
        signal.istft(overwrite=False)
        self.assertTrue(not np.any(signal.audio_data - self.sine_wave)) # check if all are exactly 0
        signal.istft()
        # they should not be exactly zero at this point, because overwrite is on
        self.assertTrue(not np.any(signal.get_channel(1)[0:len(self.sine_wave)] - self.sine_wave))

        # make sure these don't crash or nothing silly
        signal.stft(use_librosa=True)
        signal.istft(use_librosa=True)

    def test_get_channel(self):
        pass

    def test_arithmetic(self):
        a = nussl.AudioSignal(self.path)
        b = nussl.AudioSignal('../Input/k0140.wav')

        with self.assertRaises(Exception):
            a.add(b)
        with self.assertRaises(Exception):
            a.sub(b)
        with self.assertRaises(Exception):
            c = a + b
        with self.assertRaises(Exception):
            c = a - b
        with self.assertRaises(Exception):
            a += b
        with self.assertRaises(Exception):
            a -= b

        self.assertTrue(np.allclose((a + a).audio_data, a.audio_data + a.audio_data))
        self.assertTrue(np.allclose((a - a).audio_data, a.audio_data - a.audio_data))


if __name__ == '__main__':
    unittest.main()
