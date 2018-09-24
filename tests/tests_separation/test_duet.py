#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import unittest
import nussl
import numpy as np
import os

from test_base.benchmark_test_base import BenchmarkTestBase


class DuetUnitTests(BenchmarkTestBase):

    @classmethod
    def setUpClass(cls):
        cls.dev1_female3 = nussl.efz_utils.download_audio_file('dev1_female3_inst_mix.wav')
        cls.dev1_wdrums = nussl.efz_utils.download_audio_file('dev1_wdrums_inst_mix.wav')
        cls.signal = nussl.AudioSignal(cls.dev1_female3)

        cls.benchmark_dict = DuetUnitTests.load_benchmarks()

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.dev1_female3)
        os.remove(cls.dev1_wdrums)
        DuetUnitTests.remove_benchmarks()

    def test_multiple_duet(self):
        benchmark_mask = self.benchmark_dict['benchmark_masks']
        duet = nussl.Duet(self.signal, 3)
        duet.run()
        duet.audio_signal = nussl.AudioSignal(self.dev1_wdrums)
        duet.run()
        duet.audio_signal = nussl.AudioSignal(self.dev1_female3)
        duet_masks = duet.run()
        for i in range(len(duet_masks)):
            assert np.array_equal(benchmark_mask[i].mask, duet_masks[i].mask)

    def test_duet_final_outputs(self):
        # Test final outputs
        benchmark_mask = self.benchmark_dict['benchmark_masks']

        duet = nussl.Duet(self.signal, 3)
        duet_masks = duet.run()
        for i in range(len(duet_masks)):
            assert np.array_equal(benchmark_mask[i].mask, duet_masks[i].mask)

    def test_compute_spectrogram_1_channel(self):
        # Test with one channel, should throw value error
        num_samples = 100  # 1 second
        np_sin = np.sin(np.linspace(0, 100 * 2 * np.pi, num_samples))  # Freq = 100 Hz
        signal = nussl.AudioSignal(audio_data_array=np_sin)
        with self.assertRaises(ValueError):
            duet = nussl.Duet(signal, 3)
            duet._compute_spectrogram(duet.sample_rate)

    def test_compute_spectrogram_wmat(self):
        # Load Duet values to benchmark against
        duet = nussl.Duet(self.signal, 3)
        duet_sft0, duet_sft1, duet_wmat = duet._compute_spectrogram(duet.sample_rate)
        assert np.allclose(self.benchmark_dict['benchmark_stft_ch0'], duet_sft0)
        assert np.allclose(self.benchmark_dict['benchmark_stft_ch1'], duet_sft1)
        assert np.allclose(self.benchmark_dict['benchmark_wmat'], duet_wmat)

    def test_compute_atn_delay(self):
        # Use the same stfts for comparing the two functions' outputs
        duet = nussl.Duet(self.signal, 3)
        duet.stft_ch0 = self.benchmark_dict['benchmark_stft_ch0']
        duet.stft_ch1 = self.benchmark_dict['benchmark_stft_ch1']
        duet.frequency_matrix = self.benchmark_dict['benchmark_wmat']

        symmetric_atn, delay = duet._compute_atn_delay(duet.stft_ch0, duet.stft_ch1,
                                                       duet.frequency_matrix)

        assert np.allclose(self.benchmark_dict['benchmark_sym_atn'], symmetric_atn)
        assert np.allclose(self.benchmark_dict['benchmark_delay'], delay)

    def test_make_histogram(self):
        # Use the same stfts for comparing this function's outputs
        duet = nussl.Duet(self.signal, 3)

        # Set these matrices as KNOWN ground truth values up til this point
        duet.stft_ch0 = self.benchmark_dict['benchmark_stft_ch0']
        duet.stft_ch1 = self.benchmark_dict['benchmark_stft_ch1']
        duet.frequency_matrix = self.benchmark_dict['benchmark_wmat']
        duet.symmetric_atn = self.benchmark_dict['benchmark_sym_atn']
        duet.delay = self.benchmark_dict['benchmark_delay']

        # Test against these matrices
        benchmark_hist = self.benchmark_dict['benchmark_hist']
        benchmark_atn_bins = self.benchmark_dict['benchmark_atn_bins']
        benchmark_delay_bins = self.benchmark_dict['benchmark_delay_bins']

        # This is the calculation we are testing against
        hist, atn_bins, delay_bins = duet._make_histogram()

        assert np.allclose(benchmark_hist, hist)
        assert np.all(benchmark_atn_bins == atn_bins)
        assert np.all(benchmark_delay_bins == delay_bins)

    def test_peak_indices(self):
        duet = nussl.Duet(self.signal, 3)

        # Set these matrices as KNOWN ground truth values up til this point
        duet.stft_ch0 = self.benchmark_dict['benchmark_stft_ch0']
        duet.stft_ch1 = self.benchmark_dict['benchmark_stft_ch1']
        duet.frequency_matrix = self.benchmark_dict['benchmark_wmat']
        duet.symmetric_atn = self.benchmark_dict['benchmark_sym_atn']
        duet.delay = self.benchmark_dict['benchmark_delay']

        # Test against these matrices
        benchmark_hist = self.benchmark_dict['benchmark_hist']
        benchmark_peak_indices = self.benchmark_dict['benchmark_peak_indices']

        # This is the calculation we are testing against
        duet_peak_indices = nussl.utils.find_peak_indices(benchmark_hist, duet.num_sources,
                                                          threshold=duet.peak_threshold,
                                                          min_dist=[duet.attenuation_min_distance,
                                                                    duet.delay_min_distance])

        assert np.all(benchmark_peak_indices == duet_peak_indices)

    def test_convert_peaks(self):
        duet = nussl.Duet(self.signal, 3)

        # Set these matrices as KNOWN ground truth values up til this point
        duet.stft_ch0 = self.benchmark_dict['benchmark_stft_ch0']
        duet.stft_ch1 = self.benchmark_dict['benchmark_stft_ch1']
        duet.frequency_matrix = self.benchmark_dict['benchmark_wmat']
        duet.symmetric_atn = self.benchmark_dict['benchmark_sym_atn']
        duet.delay = self.benchmark_dict['benchmark_delay']
        duet.attenuation_bins = self.benchmark_dict['benchmark_atn_bins']
        duet.delay_bins = self.benchmark_dict['benchmark_delay_bins']
        duet.peak_indices = self.benchmark_dict['benchmark_peak_indices']

        # Test against these matrices
        benchmark_delay_peak = self.benchmark_dict['benchmark_delay_peak']
        benchmark_atn_peak = self.benchmark_dict['benchmark_atn_peak']
        benchmark_atn_delay_est = self.benchmark_dict['benchmark_atn_delay_est']

        # This is the calculation we are testing against
        delay_peak, atn_delay_est, atn_peak = duet._convert_peaks(duet.peak_indices)

        assert np.all(benchmark_delay_peak == delay_peak)
        assert np.all(benchmark_atn_delay_est == atn_delay_est)
        assert np.all(benchmark_atn_peak == atn_peak)

    def test_compute_masks(self):
        duet = nussl.Duet(self.signal, 3)

        # Set these matrices as KNOWN ground truth values up til this point
        duet.stft_ch0 = self.benchmark_dict['benchmark_stft_ch0']
        duet.stft_ch1 = self.benchmark_dict['benchmark_stft_ch1']
        duet.frequency_matrix = self.benchmark_dict['benchmark_wmat']
        duet.symmetric_atn = self.benchmark_dict['benchmark_sym_atn']
        duet.delay = self.benchmark_dict['benchmark_delay']
        duet.attenuation_bins = self.benchmark_dict['benchmark_atn_bins']
        duet.delay_bins = self.benchmark_dict['benchmark_delay_bins']
        duet.peak_indices = self.benchmark_dict['benchmark_peak_indices']
        duet.delay_peak = self.benchmark_dict['benchmark_delay_peak']
        duet.atn_peak = self.benchmark_dict['benchmark_atn_peak']

        # Test against these matrices
        benchmark_masks = self.benchmark_dict['benchmark_masks']

        # This is the calculation we are testing against
        masks = duet._compute_masks()
        for i in range(len(masks)):
            assert np.array_equal(benchmark_masks[i].mask, masks[i].mask)

    @unittest.skip('Broken - AudioSignal API changes')
    def test_make_audio_signals(self):
        duet = nussl.Duet(self.signal, 3)
        duet.stft_ch0 = self.benchmark_dict['benchmark_stft_ch0']
        duet.stft_ch1 = self.benchmark_dict['benchmark_stft_ch1']
        duet.frequency_matrix = self.benchmark_dict['benchmark_wmat']
        duet.symmetric_atn = self.benchmark_dict['benchmark_sym_atn']
        duet.delay = self.benchmark_dict['benchmark_delay']
        duet.atn_bins = self.benchmark_dict['benchmark_atn_bins']
        duet.delay_bins = self.benchmark_dict['benchmark_delay_bins']
        duet.peak_indices = self.benchmark_dict['benchmark_peak_indices']
        duet.delay_peak = self.benchmark_dict['benchmark_delay_peak']
        duet.atn_peak = self.benchmark_dict['benchmark_atn_peak']
        duet.result_masks = self.benchmark_dict['benchmark_masks']

        final_signals_path = os.path.join('duet_reference', 'duet_benchmarks',
                                          'benchmark_final_signals.npy')
        benchmark_final_signals = np.load(final_signals_path)

        final_signals = duet.make_audio_signals()

        # Is the audio data the same?
        assert all(np.array_equal(benchmark_final_signals[i].audio_data,
                                  final_signals[i].audio_data)
                   for i in range(len(final_signals)))

        # Check to see if AudioSignal's API changed; do we need to refreeze?
        assert all(benchmark_final_signals[i] == final_signals[i]
                   for i in range(len(final_signals)))

        assert np.all(benchmark_final_signals == final_signals)

        # test_smooth_matrix

        # TODO: fix this function before writing test for it

