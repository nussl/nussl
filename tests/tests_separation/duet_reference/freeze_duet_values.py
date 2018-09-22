#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import nussl
import numpy as np
import os


def freeze_duet_values():
    """
    Freezes essential values from DUET in its current implementation for benchmarking
    See test_benchmark_duet() in test_duet.py for usage
    """
    signal = nussl.AudioSignal(nussl.efz_utils.download_audio_file('dev1_female3_inst_mix.wav'))
    duet = nussl.Duet(signal, 3)
    output_folder = os.path.abspath('duet_benchmarks')
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    duet.stft_ch0, duet.stft_ch1, \
    duet.frequency_matrix = duet._compute_spectrogram(duet.sample_rate)
    np.save(os.path.join(output_folder, 'benchmark_stft_ch0'), duet.stft_ch0)
    np.save(os.path.join(output_folder, 'benchmark_stft_ch1'), duet.stft_ch1)
    np.save(os.path.join(output_folder, 'benchmark_wmat'), duet.frequency_matrix)

    duet.symmetric_atn, duet.delay = duet._compute_atn_delay(duet.stft_ch0, duet.stft_ch1,
                                                             duet.frequency_matrix)
    np.save(os.path.join(output_folder, 'benchmark_sym_atn'), duet.symmetric_atn)
    np.save(os.path.join(output_folder, 'benchmark_delay'), duet.delay)

    duet.normalized_attenuation_delay_histogram, \
    duet.attenuation_bins, duet.delay_bins = duet._make_histogram()
    np.save(os.path.join(output_folder, 'benchmark_hist'),
            duet.normalized_attenuation_delay_histogram)
    np.save(os.path.join(output_folder, 'benchmark_atn_bins'), duet.attenuation_bins)
    np.save(os.path.join(output_folder, 'benchmark_delay_bins'), duet.delay_bins)

    duet.peak_indices = nussl.utils.find_peak_indices(duet.normalized_attenuation_delay_histogram,
                                                      duet.num_sources,
                                                      threshold=duet.peak_threshold,
                                                      min_dist=[duet.attenuation_min_distance,
                                                                duet.delay_min_distance])
    np.save(os.path.join(output_folder, 'benchmark_peak_indices'), duet.peak_indices)

    duet.delay_peak, duet.atn_delay_est, duet.atn_peak = duet._convert_peaks(duet.peak_indices)
    np.save(os.path.join(output_folder, 'benchmark_delay_peak'), duet.delay_peak)
    np.save(os.path.join(output_folder, 'benchmark_atn_delay_est'), duet.atn_delay_est)
    np.save(os.path.join(output_folder, 'benchmark_atn_peak'), duet.atn_peak)

    duet.masks = duet._compute_masks()
    np.save(os.path.join(output_folder, 'benchmark_masks'), duet.masks)

    final_signals = duet.make_audio_signals()
    np.save(os.path.join(output_folder, 'benchmark_final_signals'), final_signals)


if __name__ == '__main__':
    freeze_duet_values()
