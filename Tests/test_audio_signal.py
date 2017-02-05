from __future__ import division

import unittest
import nussl
import numpy as np
import scipy.io.wavfile as wav
import os
import warnings


class AudioSignalUnitTests(unittest.TestCase):
    sr = nussl.constants.DEFAULT_SAMPLE_RATE
    dur = 3  # seconds
    length = dur * sr

    def setUp(self):
        input_folder = os.path.join('..', 'Input')
        output_folder = os.path.join('..', 'Output')
        ext = '.wav'
        self.all_inputs = [os.path.join(input_folder, f) for f in os.listdir(input_folder)
                           if os.path.splitext(f)[1] == ext]

        self.input_path1 = os.path.join(input_folder, 'k0140_int.wav')
        self.input_path2 = os.path.join(input_folder, 'k0140.wav')

        self.out_path1 = os.path.join(output_folder, 'k0140_int_output.wav')
        self.out_path2 = os.path.join(output_folder, 'k0140_output.wav')

    def test_load(self):
        # Load from file
        a = nussl.AudioSignal(self.input_path1)
        b = nussl.AudioSignal()
        b.load_audio_from_file(self.input_path1)

        assert (np.array_equal(a.audio_data, b.audio_data))
        assert (a.sample_rate == b.sample_rate)

        # Load from array
        ref_sr, ref_data = wav.read(self.input_path1)
        c = nussl.AudioSignal(audio_data_array=ref_data, sample_rate=ref_sr)

        with self.assertRaises(ValueError):
            nussl.AudioSignal(self.input_path1, ref_data)
        with self.assertRaises(ValueError):
            nussl.AudioSignal(path_to_input_file=self.input_path1, audio_data_array=ref_data)

        d = nussl.AudioSignal()
        d.load_audio_from_array(ref_data, ref_sr)

        assert (np.array_equal(c.audio_data, d.audio_data))
        assert (c.sample_rate == d.sample_rate)
        assert (b.sample_rate == c.sample_rate)
        assert (np.array_equal(b.audio_data, c.audio_data))

    def test_load_audio_from_file(self):

        # Do some preliminary checks
        signal_info = []
        for path in self.all_inputs:
            ref_sr, ref_data = wav.read(path)
            ref_dur = len(ref_data) / ref_sr
            n_chan = 1 if len(ref_data.shape) == 1 else ref_data.shape[1]
            signal_info.append({'duration' : ref_dur,
                                'sample_rate' : ref_sr,
                                'length' : len(ref_data),
                                'n_chan': n_chan})

            a = nussl.AudioSignal()
            a.load_audio_from_file(path)

            assert a.signal_length == len(ref_data)
            assert a.num_channels == n_chan
            assert a.sample_rate == ref_sr
            assert np.isclose(a.signal_duration, ref_dur)
            assert a.active_region_is_default

        # Test different offsets
        percentages = [0.1, 0.25, 0.4, 0.5, 0.75, 0.9]
        for i, path in enumerate(self.all_inputs):
            for start in percentages:
                offset = start * signal_info[i]['duration']
                ref_length = int(round(signal_info[i]['length'] - offset * signal_info[i]['sample_rate']))

                a = nussl.AudioSignal()
                a.load_audio_from_file(path, offset=offset)

                assert abs(a.signal_length - ref_length) <= 1  # Sometimes ref_length is off by 1 due to rounding

        # Test different durations
        for i, path in enumerate(self.all_inputs):
            for start in percentages:
                duration = start * signal_info[i]['duration']
                ref_length = int(round(duration * signal_info[i]['sample_rate']))

                a = nussl.AudioSignal()
                a.load_audio_from_file(path, duration=duration)

                assert abs(a.signal_length - ref_length) <= 1  # Sometimes ref_length is off by 1 due to rounding

        # Test offsets and durations
        percentages = np.arange(0.0, 0.51, 0.05)
        for i, path in enumerate(self.all_inputs):
            for start in percentages:
                for duration in percentages:
                    offset = start * signal_info[i]['duration']
                    duration = duration * signal_info[i]['duration']
                    ref_length = int(round(duration * signal_info[i]['sample_rate']))

                    a = nussl.AudioSignal()
                    a.load_audio_from_file(path, offset=offset, duration=duration)

                    assert abs(a.signal_length - ref_length) <= 1  # Sometimes ref_length is off by 1 due to rounding

        # Test error cases
        path = self.input_path1
        sr, data = wav.read(path)
        dur = len(data) / sr
        with self.assertRaises(ValueError):
            offset = dur + 1.0
            a = nussl.AudioSignal()
            a.load_audio_from_file(path, offset=offset)

        with self.assertRaises(IOError):
            a = nussl.AudioSignal()
            a.load_audio_from_file('not a real path')

        # Make sure this is okay
        offset = dur / 2.0
        duration = dur
        a = nussl.AudioSignal()
        a.load_audio_from_file(path, offset=offset, duration=duration)

    def test_write_to_file_path1(self):
        a = nussl.AudioSignal(self.input_path1)
        a.write_audio_to_file(self.out_path1)
        b = nussl.AudioSignal(self.out_path1)

        assert (a.sample_rate == b.sample_rate)
        assert (np.allclose(a.audio_data, b.audio_data))

    def test_write_to_file_path2(self):
        a = nussl.AudioSignal()
        a.load_audio_from_file(self.input_path1)
        a.write_audio_to_file(self.out_path1)
        b = nussl.AudioSignal(self.out_path1)

        assert (a.sample_rate == b.sample_rate)
        assert (np.allclose(a.audio_data, b.audio_data))

    def test_write_to_file_array1(self):
        sr, data = wav.read(self.input_path1)
        a = nussl.AudioSignal(audio_data_array=data, sample_rate=sr)
        a.write_audio_to_file(self.out_path1)
        b = nussl.AudioSignal(self.out_path1)

        assert (a.sample_rate == b.sample_rate)
        assert (np.allclose(a.audio_data, b.audio_data))

    def test_write_to_file_array2(self):
        sr, data = wav.read(self.input_path1)
        a = nussl.AudioSignal()
        a.load_audio_from_array(data, sr)
        a.write_audio_to_file(self.out_path1)
        b = nussl.AudioSignal(self.out_path1)

        assert (a.sample_rate == b.sample_rate)
        assert (np.allclose(a.audio_data, b.audio_data))

    def test_write_sample_rate(self):
        a = nussl.AudioSignal(self.input_path1)
        sample_rate = a.sample_rate // 2
        a.write_audio_to_file(self.out_path1, sample_rate=sample_rate)
        b = nussl.AudioSignal(self.out_path1)

        assert (b.sample_rate == sample_rate)

    freq = 30
    sine_wave = np.sin(np.linspace(0, freq * 2 * np.pi, length))

    def test_stft_istft_simple1(self):
        """
        Tests to make sure stft and istft don't crash with default settings.
        Returns:

        """
        a = nussl.AudioSignal(audio_data_array=self.sine_wave)
        a.stft()
        a.istft()

    def test_stft_istft_simple2(self):
        a = nussl.AudioSignal(audio_data_array=self.sine_wave)

        a.stft(use_librosa=True)
        calc_sine = a.istft(overwrite=False, use_librosa=True)

        assert np.allclose(a.audio_data, calc_sine, atol=1e-3)

        a = nussl.AudioSignal(audio_data_array=self.sine_wave)
        a.stft(use_librosa=False)
        calc_sine = a.istft(overwrite=False, use_librosa=False)

        assert np.allclose(a.audio_data, calc_sine)

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
        a = nussl.AudioSignal(self.input_path1)
        b = nussl.AudioSignal(self.input_path2)

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
