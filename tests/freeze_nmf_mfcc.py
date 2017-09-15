#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import nussl
import os


# Freezes essential values from NMF MFCC in its current implementation for benchmarking
# See test_benchmark_nmf_mfcc() in test_nmf_mfcc.py for usage
def main():
    path_to_benchmark_file = os.path.join('..', 'Input', 'piano_and_synth_arp_chord_mono.wav')
    signal = nussl.AudioSignal(path_to_benchmark_file)

    # Set random seed in NMF and KMeans to 0
    nmf_mfcc = nussl.NMF_MFCC(signal, num_sources=2, num_templates=6, distance_measure="euclidean",
                              num_iterations=100, random_seed=0)
    output_folder = os.path.abspath('nmf_mfcc_reference/new')

    nmf_mfcc.run()
    np.save(os.path.join(output_folder, "benchmark_labeled_templates"), nmf_mfcc.labeled_templates)
    np.save(os.path.join(output_folder, "benchmark_masks"), nmf_mfcc.masks)

    nmf_mfcc.make_audio_signals()
    np.save(os.path.join(output_folder, "benchmark_sources"), nmf_mfcc.sources)

if __name__ == '__main__':
    main()
