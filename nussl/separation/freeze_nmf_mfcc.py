#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import nussl
import os

#Freeze key NMF MFCC values to the benchmark folder
def freeze_nmf_mfcc_values():
    path_to_benchmark_file = os.path.join('..', 'Input', 'piano_and_synth_arp_chord_mono.wav')
    signal = nussl.AudioSignal(path_to_benchmark_file)
    nmf_mfcc = nussl.NMF_MFCC(signal, num_sources=2, num_templates=6, distance_measure="euclidean",
                              num_iterations=150)
    output_folder = os.path.abspath('nmf_mfcc_reference/nmf_mfcc_benchmarks')

    nmf_mfcc.run()
    np.save(os.path.join(output_folder, "benchmark_activation_matrix"), nmf_mfcc.activation_matrix)
    np.save(os.path.join(output_folder, "benchmark_templates_matrix"), nmf_mfcc.templates_matrix)
    np.save(os.path.join(output_folder, "benchmark_labeled_templates"), nmf_mfcc.labeled_templates)
    np.save(os.path.join(output_folder, "benchmark_masks"), nmf_mfcc.masks)

    nmf_mfcc.make_audio_signals()
    np.save(os.path.join(output_folder, "benchmark_sources"), nmf_mfcc.sources)

if __name__ == '__main__':
    freeze_nmf_mfcc_values()
