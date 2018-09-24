#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import unittest

import os
import copy

import numpy as np
import scipy.io.wavfile as wav
import librosa

import nussl


class AudioSignalUnitTests(unittest.TestCase):
    sr = nussl.DEFAULT_SAMPLE_RATE
    dur = 3  # seconds
    length = dur * sr

    def setUp(self):
        self.audio_input1 = nussl.efz_utils.download_audio_file('K0140.wav')
        self.audio_input2 = nussl.efz_utils.download_audio_file('K0149.wav')
        self.audio_input3 = nussl.efz_utils.download_audio_file('dev1_female3_inst_mix.wav')
        self.all_inputs = [self.audio_input1, self.audio_input2, self.audio_input3]

        self.audio_output = 'k0140_int_output.wav'
        self.png_output = 'test_graph.png'
        self.all_outputs = [self.audio_output, self.png_output]

    def tearDown(self):
        for f in self.all_outputs:
            if os.path.isfile(f):
                os.remove(f)

    def test_load(self):
        # Load from file
        a = nussl.AudioSignal(self.audio_input1)
        b = nussl.AudioSignal()
        b.load_audio_from_file(self.audio_input1)

        assert (np.array_equal(a.audio_data, b.audio_data))
        assert (a.sample_rate == b.sample_rate)

        # Load from array
        ref_sr, ref_data = wav.read(self.audio_input1)
        c = nussl.AudioSignal(audio_data_array=ref_data, sample_rate=ref_sr)

        with self.assertRaises(Exception):
            nussl.AudioSignal(self.audio_input1, ref_data)
        with self.assertRaises(Exception):
            nussl.AudioSignal(path_to_input_file=self.audio_input1, audio_data_array=ref_data)

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
            signal_info.append({'duration': ref_dur,
                                'sample_rate': ref_sr,
                                'length': len(ref_data),
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
                ref_length = int(round(signal_info[i]['length'] - offset *
                                       signal_info[i]['sample_rate']))

                a = nussl.AudioSignal()
                a.load_audio_from_file(path, offset=offset)

                # Sometimes ref_length is off by 1 due to rounding
                assert abs(a.signal_length - ref_length) <= 1

        # Test different durations
        for i, path in enumerate(self.all_inputs):
            for start in percentages:
                duration = start * signal_info[i]['duration']
                ref_length = int(round(duration * signal_info[i]['sample_rate']))

                a = nussl.AudioSignal()
                a.load_audio_from_file(path, duration=duration)

                # Sometimes ref_length is off by 1 due to rounding
                assert abs(a.signal_length - ref_length) <= 1

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

                    # Sometimes ref_length is off by 2 due to rounding
                    assert abs(a.signal_length - ref_length) <= 2

        # Test error cases
        path = self.audio_input1
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
        a = nussl.AudioSignal(self.audio_input1)
        a.write_audio_to_file(self.audio_output)
        b = nussl.AudioSignal(self.audio_output)

        assert (a.sample_rate == b.sample_rate)
        assert (np.allclose(a.audio_data, b.audio_data))

    def test_write_to_file_path2(self):
        a = nussl.AudioSignal()
        a.load_audio_from_file(self.audio_input1)
        a.write_audio_to_file(self.audio_output)
        b = nussl.AudioSignal(self.audio_output)

        assert (a.sample_rate == b.sample_rate)
        assert (np.allclose(a.audio_data, b.audio_data))

    def test_write_to_file_array1(self):
        sr, data = wav.read(self.audio_input1)
        a = nussl.AudioSignal(audio_data_array=data, sample_rate=sr)
        a.write_audio_to_file(self.audio_output)
        b = nussl.AudioSignal(self.audio_output)

        assert (a.sample_rate == b.sample_rate)
        assert (np.allclose(a.audio_data, b.audio_data))

    def test_write_to_file_array2(self):
        sr, data = wav.read(self.audio_input1)
        a = nussl.AudioSignal()
        a.load_audio_from_array(data, sr)
        a.write_audio_to_file(self.audio_output)
        b = nussl.AudioSignal(self.audio_output)

        assert (a.sample_rate == b.sample_rate)
        assert (np.allclose(a.audio_data, b.audio_data))

    def test_write_sample_rate(self):
        a = nussl.AudioSignal(self.audio_input1)
        sample_rate = a.sample_rate // 2
        a.write_audio_to_file(self.audio_output, sample_rate=sample_rate)
        b = nussl.AudioSignal(self.audio_output)

        assert (b.sample_rate == sample_rate)

    freq = 30
    sine_wave = np.sin(np.linspace(0, freq * 2 * np.pi, length))

    def test_resample(self):
        # Check that sample rate property changes
        a = nussl.AudioSignal(self.audio_input1)
        b = nussl.AudioSignal(self.audio_input1)
        b.resample(a.sample_rate / 2)
        assert (b.sample_rate == a.sample_rate/2)

    def test_resample_on_load_from_file(self):
        # Test resample right when loading from file vs resampling after loading
        a = nussl.AudioSignal(self.audio_input1)
        a.resample(48000)
        b = nussl.AudioSignal()
        b.load_audio_from_file(self.audio_input1, new_sample_rate=48000)
        assert (a.sample_rate == b.sample_rate)
        assert (np.allclose(a.audio_data, b.audio_data))

    def test_resample_vs_librosa_load(self):
        # Check against librosa load function
        a = nussl.AudioSignal(self.audio_input1)
        a.resample(48000)
        b_audio_data, b_sample_rate = librosa.load(self.audio_input1, sr=48000)
        assert (a.sample_rate == b_sample_rate)
        assert (np.allclose(a.audio_data, b_audio_data))

    def test_default_sr_on_load_from_array(self):
        # Check that the default sample rate is set when no sample rate is provided load_audio_from_array
        sr, data = wav.read(self.audio_input1)
        a = nussl.AudioSignal()
        a.load_audio_from_array(data)
        assert(a.sample_rate == nussl.DEFAULT_SAMPLE_RATE)

    def test_sr_on_load_from_array(self):
        # Check that the passed in sample rate is being set in load_audio_from_array
        a = nussl.AudioSignal(self.audio_input1)
        sr, data = wav.read(self.audio_input1)
        b = nussl.AudioSignal()
        b.load_audio_from_array(data, sample_rate=sr)
        assert (a.sample_rate == b.sample_rate)
        assert (np.allclose(a.audio_data, b.audio_data))

    def test_plot_time_domain_stereo(self):
        # Stereo signal that should plot both channels on same plot
        a = nussl.AudioSignal(self.audio_input3)
        a.plot_time_domain()

    def test_plot_time_domain_specific_channel(self):
        # Stereo signal that should only plot the specified channel
        a = nussl.AudioSignal(self.audio_input3)
        a.plot_time_domain(channel=0)

    def test_plot_time_domain_mono(self):
        # Mono signal plotting
        a = nussl.AudioSignal(self.audio_input3)
        a.to_mono(overwrite=True)
        a.plot_time_domain()

    def test_plot_time_domain_multi_channel(self):
        # Plotting a signal with 5 channels
        num_test_channels = 5
        freq_multiple = 5
        freqs = [i * freq_multiple for i in range(1, num_test_channels+1)]
        test_signal = np.array([np.sin(np.linspace(0, i * 2 * np.pi, self.length)) for i in freqs[:num_test_channels]])
        a = nussl.AudioSignal(audio_data_array=test_signal)
        a.plot_time_domain()

    def test_plot_time_domain_sample_on_xaxis(self):
        # Plotting a stereo signal with sample numbers on the x axis instead of time
        a = nussl.AudioSignal(self.audio_input3)
        a.plot_time_domain(x_label_time=False)

    def test_plot_time_domain_save_to_path_and_rename(self):
        # Plotting and saving the plot with a new name to a folder
        a = nussl.AudioSignal(self.audio_input3)
        a.plot_time_domain(file_path_name=self.png_output)

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

    def test_to_mono_channel_dimension(self):
        """
        Test functionality and correctness of AudioSignal.to_mono() function.
        Returns:

        """
        # Load input file
        signal = nussl.AudioSignal(path_to_input_file=self.audio_input3)
        signal.stft_params = signal.stft_params
        signal_stft = signal.stft()
        assert (signal_stft.shape[nussl.STFT_CHAN_INDEX] == 2)

        signal.to_mono(overwrite=True)
        signal.stft_params = signal.stft_params
        signal_stft = signal.stft()
        assert (signal_stft.shape[nussl.STFT_CHAN_INDEX]== 1)

    def test_stft(self):
        """
        Test some basic functionality of the STFT interface for the AudioSignal object
        All verification of the STFT calculation is done in test_spectral_utils file.
        """
        signal = nussl.AudioSignal(audio_data_array=self.sine_wave)

        self.assertTrue(signal.stft_data is None)
        signal.stft(overwrite=False, use_librosa=False)
        self.assertTrue(signal.stft_data is None)
        _ = signal.stft()
        self.assertTrue(signal.stft_data is not None)
        signal.istft(overwrite=False, use_librosa=False)

        # check if all are exactly 0
        self.assertTrue(not np.any(signal.audio_data - self.sine_wave))
        signal.istft()

        # Make sure they're the same length
        self.assertTrue(len(signal) == len(self.sine_wave))

        # They should not be exactly zero at this point, because overwrite is on
        self.assertTrue(np.any(signal.get_channel(0)[0:len(self.sine_wave)] - self.sine_wave))

        # make sure these don't crash or nothing silly
        signal.stft(use_librosa=True)
        signal.istft(use_librosa=True)

    def test_get_channel(self):
        # Here we're setting up signals with 1 to 8 channels
        # Each channel has a sine wave of different frequency in it

        # This is the frequencies for our different channels
        max_n_channels = 8
        freq_multiple = 300
        freqs = [i * freq_multiple for i in range(max_n_channels)]

        # Make the signals and test
        for f in range(1, max_n_channels):  # 1-8 channel mixtures
            sig = np.array([np.sin(np.linspace(0, i * 2 * np.pi, self.length)) for i in freqs[:f]])
            self._get_channel_helper(sig, len(sig))

    def _get_channel_helper(self, signal, n_channels):
        a = nussl.AudioSignal(audio_data_array=signal)

        # Check that we are counting our channels correctly
        assert a.num_channels == n_channels

        # Check that we can get every channel with AudioSignal.get_channel()
        for i, ch in enumerate(signal):
            assert np.array_equal(a.get_channel(i), ch)

        # Check that attempting to get higher channels raises exception
        for i in range(n_channels, n_channels + 10):
            with self.assertRaises(ValueError):
                a.get_channel(i)

        # Check that attempting to get lower channels raises exception
        for i in range(-1, -11, -1):
            with self.assertRaises(ValueError):
                a.get_channel(i)

        # Check that AudioSignal.get_channels() generator works
        i = 0
        for ch in a.get_channels():
            assert np.array_equal(ch, signal[i, :])
            i += 1
        assert i == a.num_channels

    def test_arithmetic(self):

        # These signals have two different lengths
        a = nussl.AudioSignal(self.audio_input1)
        b = nussl.AudioSignal(self.audio_input2)

        with self.assertRaises(Exception):
            a.add(b)
        with self.assertRaises(Exception):
            a.subtract(b)
        with self.assertRaises(Exception):
            _ = a + b
        with self.assertRaises(Exception):
            _ = a - b
        with self.assertRaises(Exception):
            a += b
        with self.assertRaises(Exception):
            a -= b
        with self.assertRaises(Exception):
            _ = a * b
        with self.assertRaises(Exception):
            _ = a / b
        with self.assertRaises(Exception):
            _ = a / a

        self.assertTrue(np.allclose((a + a).audio_data, a.audio_data + a.audio_data))
        self.assertTrue(np.allclose((a - a).audio_data, a.audio_data - a.audio_data))

        c = a * 2
        self.assertTrue(np.allclose(c.audio_data, a.audio_data * 2))
        d = copy.copy(a)
        d *= 2
        self.assertTrue(np.allclose(c.audio_data, d.audio_data))

        c = a / 2
        self.assertTrue(np.allclose(c.audio_data, a.audio_data / 2))
        d = copy.copy(a)
        d /= 2
        self.assertTrue(np.allclose(c.audio_data, d.audio_data))




if __name__ == '__main__':
    unittest.main()
