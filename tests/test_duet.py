#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import unittest
import nussl
import numpy as np
import os


class DuetUnitTests(unittest.TestCase):

    # Update this if the benchmark file changes and rerun freeze_duet_values() (below)
    path_to_benchmark_file = os.path.join('..', 'Input', 'dev1_female3_inst_mix.wav')

    def setUp(self):
        self.signal = nussl.AudioSignal(self.path_to_benchmark_file)
        # Call benchmarks
        self.benchmark_dict = self.load_benchmarks()

    @staticmethod
    def load_benchmarks():
        benchmark_dict = {}
        directory = 'duet_reference/duet_benchmarks'
        for filename in os.listdir(directory):
            key = os.path.splitext(filename)[0]
            file_path = os.path.join('duet_reference', 'duet_benchmarks', filename)
            value = np.load(file_path)
            benchmark_dict[key] = value
        return benchmark_dict

    def test_multiple_duet(self):
        mask_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_masks.npy')
        benchmark_mask = np.load(mask_path)
        duet = nussl.Duet(self.signal, 3)
        duet_masks = duet.run()
        path_to_benchmark_file = os.path.join('..', 'Input', 'dev1_wdrums_inst_mix.wav')
        duet.audio_signal = nussl.AudioSignal(path_to_benchmark_file)
        duet_masks = duet.run()
        path_to_benchmark_file = os.path.join('..', 'Input', 'dev1_female3_inst_mix.wav')
        duet.audio_signal = nussl.AudioSignal(path_to_benchmark_file)
        duet_masks = duet.run()
        for i in range(len(duet_masks)):
            assert np.array_equal(benchmark_mask[i].mask, duet_masks[i].mask)

    def test_duet_final_outputs(self):
        # Test final outputs
        mask_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_masks.npy')
        benchmark_mask = np.load(mask_path)

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

        symmetric_atn, delay = duet._compute_atn_delay(duet.stft_ch0, duet.stft_ch1, duet.frequency_matrix)

        assert np.allclose(self.benchmark_dict['benchmark_sym_atn'], symmetric_atn)
        assert np.allclose(self.benchmark_dict['benchmark_delay'], delay)

    def test_make_histogram(self):
        # Use the same stfts for comparing this function's outputs
        duet = nussl.Duet(self.signal, 3)
        duet.stft_ch0 = self.benchmark_dict['benchmark_stft_ch0']
        duet.stft_ch1 = self.benchmark_dict['benchmark_stft_ch1']
        duet.frequency_matrix = self.benchmark_dict['benchmark_wmat']
        duet.symmetric_atn = self.benchmark_dict['benchmark_sym_atn']
        duet.delay = self.benchmark_dict['benchmark_delay']

        # Load benchmarks
        hist_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_hist.npy')
        atn_bins_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_atn_bins.npy')
        delay_bins_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_delay_bins.npy')
        benchmark_hist = np.load(hist_path)
        benchmark_atn_bins = np.load(atn_bins_path)
        benchmark_delay_bins = np.load(delay_bins_path)

        hist, atn_bins, delay_bins = duet._make_histogram()

        assert np.allclose(benchmark_hist, hist)
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

        duet_peak_indices = nussl.utils.find_peak_indices(benchmark_hist, duet.num_sources,
                                                          threshold=duet.peak_threshold,
                                                          min_dist=[duet.attenuation_min_distance,
                                                                    duet.delay_min_distance])

        assert np.all(benchmark_peak_indices == duet_peak_indices)

    def test_convert_peaks(self):
        duet = nussl.Duet(self.signal, 3)
        duet.stft_ch0 = self.benchmark_dict['benchmark_stft_ch0']
        duet.stft_ch1 = self.benchmark_dict['benchmark_stft_ch1']
        duet.frequency_matrix = self.benchmark_dict['benchmark_wmat']
        duet.symmetric_atn = self.benchmark_dict['benchmark_sym_atn']
        duet.delay = self.benchmark_dict['benchmark_delay']
        duet.attenuation_bins = self.benchmark_dict['benchmark_atn_bins']
        duet.delay_bins = self.benchmark_dict['benchmark_delay_bins']
        duet.peak_indices = self.benchmark_dict['benchmark_peak_indices']

        delay_peak_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_delay_peak.npy')
        atn_peak_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_atn_peak.npy')
        atn_delay_est_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_atn_delay_est.npy')
        benchmark_delay_peak = np.load(delay_peak_path)
        benchmark_atn_peak = np.load(atn_peak_path)
        benchmark_atn_delay_est = np.load(atn_delay_est_path)

        delay_peak, atn_delay_est, atn_peak = duet._convert_peaks(duet.peak_indices)

        assert np.all(benchmark_delay_peak == delay_peak)
        assert np.all(benchmark_atn_delay_est == atn_delay_est)
        assert np.all(benchmark_atn_peak == atn_peak)

    def test_compute_masks(self):
        duet = nussl.Duet(self.signal, 3)
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

        mask_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_masks.npy')
        benchmark_mask = np.load(mask_path)

        masks = duet._compute_masks()
        for i in range(len(masks)):
            assert np.array_equal(benchmark_mask[i].mask, masks[i].mask)

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

        final_signals_path = os.path.join('duet_reference', 'duet_benchmarks', 'benchmark_final_signals.npy')
        benchmark_final_signals = np.load(final_signals_path)

        final_signals = duet.make_audio_signals()

        # Is the audio data the same?
        assert all(np.array_equal(benchmark_final_signals[i].audio_data, final_signals[i].audio_data)
                   for i in range(len(final_signals)))

        # Check to see if AudioSignal's API changed; do we need to refreeze?
        assert all(benchmark_final_signals[i] == final_signals[i] for i in range(len(final_signals)))

        assert np.all(benchmark_final_signals == final_signals)

        # test_smooth_matrix

        # TODO: fix this function before writing test for it

