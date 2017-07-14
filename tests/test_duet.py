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
        path = os.path.join('..', 'Input', 'dev1_female3_inst_mix.wav')
        self.signal = nussl.AudioSignal(path)
        #Call benchmarks
        self.duet = nussl.Duet(self.signal, 3)
        self.benchmark_dict = self.load_benchmarks()

    def load_benchmarks(self):
        benchmark_dict = {}
        directory = 'duet_reference/duet_benchmarks'
        for filename in os.listdir(directory):
            key = os.path.splitext(filename)[0]
            file_path = os.path.join('duet_reference', 'duet_benchmarks', filename)
            value = np.load(file_path)
            benchmark_dict[key] = value
        return benchmark_dict

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
        duet = nussl.Duet(self.signal, 3)
        duet_sft0, duet_sft1, duet_wmat = duet._compute_spectrogram(self.duet.sample_rate)
        assert np.all(np.array(self.benchmark_dict['benchmark_stft_ch0']) == duet_sft0)
        assert np.all(np.array(self.benchmark_dict['benchmark_stft_ch1']) == duet_sft1)
        assert np.all(np.array(self.benchmark_dict['benchmark_wmat']) == duet_wmat)


    def test_compute_atn_delay(self):
        #Use the same stfts for comparing the two functions' outputs
        duet = nussl.Duet(self.signal, 3)
        duet.stft_ch0 = self.benchmark_dict['benchmark_stft_ch0']
        duet.stft_ch1 = self.benchmark_dict['benchmark_stft_ch1']
        duet.frequency_matrix = self.benchmark_dict['benchmark_wmat']

        symmetric_atn, delay = duet._compute_atn_delay()

        assert np.all(np.array(self.benchmark_dict['benchmark_sym_atn'])== symmetric_atn)
        assert np.all(np.array(self.benchmark_dict['benchmark_delay'])== delay)

    def test_make_histogram(self):
        #Use the same stfts for comparing this function's outputs
        duet = nussl.Duet(self.signal, 3)
        duet.stft_ch0 = self.benchmark_dict['benchmark_stft_ch0']
        duet.stft_ch1 = self.benchmark_dict['benchmark_stft_ch1']
        duet.frequency_matrix = self.benchmark_dict['benchmark_wmat']
        duet.symmetric_atn = self.benchmark_dict['benchmark_sym_atn']
        duet.delay = self.benchmark_dict['benchmark_delay']

        #Load benchmarks
        hist_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_hist.npy')
        atn_bins_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_atn_bins.npy')
        delay_bins_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_delay_bins.npy')
        benchmark_hist = np.load(hist_path)
        benchmark_atn_bins = np.load(atn_bins_path)
        benchmark_delay_bins = np.load(delay_bins_path)

        hist, atn_bins, delay_bins = duet.make_histogram(duet.p, duet.q)

        assert np.all(benchmark_hist == hist)
        assert np.all(benchmark_atn_bins == atn_bins)
        assert np.all(benchmark_delay_bins == delay_bins)

    def test_peak_indices(self):
        duet = nussl.Duet(self.signal, 3)
        duet.stft_ch0 = self.benchmark_dict['benchmark_stft_ch0']
        duet.stft_ch1 = self.benchmark_dict['benchmark_stft_ch1']
        duet.frequency_matrix = self.benchmark_dict['benchmark_wmat']
        duet.symmetric_atn = self.benchmark_dict['benchmark_sym_atn']
        duet.delay = self.benchmark_dict['benchmark_delay']

        hist_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_hist.npy')
        peak_indices_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_peak_indices.npy')
        benchmark_hist = np.load(hist_path)
        benchmark_peak_indices = np.load(peak_indices_path)

        duet_peak_indices = nussl.utils.find_peak_indices(benchmark_hist, self.duet.num_sources, threshold=self.duet.peak_threshold,
                                                    min_dist=[self.duet.attenuation_min_distance, self.duet.delay_min_distance])

    # original DUET returned peaks as [[28 21 44] [24 25 24]] instead of [(28, 24), (21,25), (44,24)]
    # Convert current peaks being tested to original format
        atn_indices = [x[0] for x in duet_peak_indices]
        delay_indices = [x[1] for x in duet_peak_indices]
        duet_peak_indices = np.row_stack((atn_indices, delay_indices))

        assert np.all(benchmark_peak_indices == duet_peak_indices)

    def test_convert_peaks(self):
        pass


    # def test_compute_masks

    # def test_convert_time_domain

    # test_smooth_matrix

    #TODO: fix this function before writing test for it

if __name__ == '__main__':
    unittest.main()

