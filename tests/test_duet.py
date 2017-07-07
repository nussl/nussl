#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import unittest
import nussl
import numpy as np
import scipy.io.wavfile as wav
import scipy.io
import os
import warnings


class DuetUnitTests(unittest.TestCase):
    def setUp(self):
        # Use the same audio file that the benchmarks were run with
        path = os.path.join('..', 'Input', 'dev1_female3_inst_mix.wav')
        signal = nussl.AudioSignal(path)
        self.duet = nussl.Duet(signal, 3)

    # def test_duet_final_outputs(self):
    #     #Test final outputs
    #     source_estimates_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_source_estimates.npy')
    #     atn_delay_est_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_atn_delay_est.npy')
    #     benchmark_source_estimates = np.load(source_estimates_path)
    #     benchmark_atn_delay_est = np.load(atn_delay_est_path)
    #
    #     duet_source_estimates, duet_atn_delay_est = self.duet.run()
    #     assert benchmark_source_estimates == duet_source_estimates
    #     assert benchmark_atn_delay_est == duet_atn_delay_est

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
        stft0_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_stft_ch0.npy')
        stft1_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_stft_ch1.npy')
        wmat_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_wmat.npy')

        benchmark_stft_ch0 = np.load(stft0_path)
        benchmark_stft_ch1 = np.load(stft1_path)
        benchmark_wmat = np.load(wmat_path)

        duet_sft0, duet_sft1, duet_wmat = self.duet._compute_spectrogram(self.duet.sample_rate)
        assert np.all(benchmark_stft_ch0 == duet_sft0)
        assert np.all(benchmark_stft_ch1 == duet_sft1)
        assert np.all(benchmark_wmat == duet_wmat)

    def test_compute_atn_delay(self):
        #Use the same stfts for comparing this function's outputs
        stft0_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_stft_ch0.npy')
        stft1_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_stft_ch1.npy')
        wmat_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_wmat.npy')
        self.duet.stft_ch0 = np.load(stft0_path)
        self.duet.stft_ch1 = np.load(stft1_path)
        self.duet.frequency_matrix = np.load(wmat_path)

        #Load benchmarks
        sym_atn_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_sym_atn.npy')
        delay_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_delay.npy')
        benchmark_sym_atn = np.load(sym_atn_path)
        benchmark_delay = np.load(delay_path)

        symmetric_atn, delay = self.duet._compute_atn_delay()

        assert np.all(benchmark_sym_atn == symmetric_atn)
        assert np.all(benchmark_delay == delay)

    # def test_make_histogram(self):
    #     #Use the same stfts for comparing this function's outputs
    #     stft0_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_stft_ch0.npy')
    #     stft1_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_stft_ch1.npy')
    #     wmat_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_wmat.npy')
    #     sym_atn_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_sym_atn.npy')
    #     delay_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_delay.npy')
    #     self.duet.stft_ch0 = np.load(stft0_path)
    #     self.duet.stft_ch1 = np.load(stft1_path)
    #     self.duet.frequency_matrix = np.load(wmat_path)
    #     self.symmetric_atn = np.load(sym_atn_path)
    #     self.delay = np.load(delay_path)
    #
    #     #Load benchmarks
    #     hist_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_hist.npy')
    #     atn_bins_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_atn_bins.npy')
    #     delay_bins_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_delay_bins.npy')
    #     benchmark_hist = np.load(hist_path)
    #     benchmark_atn_bins = np.load(atn_bins_path)
    #     benchmark_delay_bins = np.load(delay_bins_path)
    #
    #     hist, atn_bins, delay_bins = self.duet.make_histogram(self.duet.p, self.duet.q)
    #
    #     assert np.all(benchmark_hist == hist)
    #     assert np.all(benchmark_atn_bins == atn_bins)
    #     assert np.all(benchmark_delay_bins == delay_bins)
    #
    # def test_peak_indices(self):
    #     hist_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_hist.npy')
    #     peak_indices_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_peak_indices.npy')
    #     benchmark_hist = np.load(hist_path)
    #     benchmark_peak_indices = np.load(peak_indices_path)
    #
    #     duet_peak_indices = nussl.utils.find_peak_indices(benchmark_hist, self.duet.num_sources, threshold=self.duet.peak_threshold,
    #                                                 min_dist=[self.duet.attenuation_min_distance, self.duet.delay_min_distance])
    #
    #     assert np.all(benchmark_peak_indices == duet_peak_indices)

    # def test_convert_peaks

    # def test_compute_masks

    # def test_convert_time_domain

    # test_smooth_matrix


if __name__ == '__main__':
    unittest.main()

