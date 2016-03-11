import unittest
import numpy as np
import nussl
import nussl
import numpy as np
import scipy.io.wavfile as wav


class TestAudioSignal(unittest.TestCase):
    sr = nussl.constants.DEFAULT_SAMPLE_RATE
    dur = 3 # seconds
    length = dur * sr
    def setUp(self):
        self.path = "../Input/k0140_int.wav"
        self.out_path = '../Output/test_write.wav'

    def test_load(self):
        # Load from file
        a = nussl.AudioSignal(self.path)
        b = nussl.AudioSignal()
        b.load_audio_from_file(self.path)

        assert(np.array_equal(a.audio_data, b.audio_data))
        assert(a.sample_rate == b.sample_rate)

        # Load from array
        sr, data = wav.read(self.path)
        c = nussl.AudioSignal(audio_data_array=data, sample_rate=sr)
        d = nussl.AudioSignal()
        d.load_audio_from_array(data, sr)

        assert(np.array_equal(c.audio_data, d.audio_data))
        assert(c.sample_rate == d.sample_rate)
        assert(b.sample_rate == c.sample_rate)
        assert(np.array_equal(b.audio_data, c.audio_data))

    def test_write_to_file_path1(self):
        a = nussl.AudioSignal(self.path)
        a.write_audio_to_file(self.out_path)
        b = nussl.AudioSignal(self.out_path)

        assert(a.sample_rate == b.sample_rate)
        assert(np.allclose(a.audio_data, b.audio_data))

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

        assert(b.sample_rate == sample_rate)

    freq = 30
    sine_wave = np.sin(np.linspace(0, freq * 2 * np.pi, length))

    def test_stft_istft_simple1(self):
        a = nussl.AudioSignal(audio_data_array=self.sine_wave)
        a.stft()
        a.istft()

    def test_stft_istft_simple2(self):
        a = nussl.AudioSignal(audio_data_array=self.sine_wave)

        # complete OLA
        a.stft_params.window_type = nussl.WindowType.RECTANGULAR
        a.stft_params.hop_length = a.stft_params.window_length

        a.stft()
        calc_sine = a.istft(overwrite=False)

        l = a.stft_params.hop_length//2
        # diff = a.audio_data[0, 0:a.signal_length] - calc_sine[0, l:a.signal_length+l]

        assert (np.allclose(a.audio_data[0, 0:a.signal_length], calc_sine[0, l:a.signal_length+l]))




    def test_stft(self):
        pass

    def test_istft(self):
        pass

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
