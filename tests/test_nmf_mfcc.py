#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import nussl
import os



class NMFMFCCUnitTests(unittest.TestCase):

    # Update this if the benchmark file changes and rerun freeze_duet_values() (below)
    path_to_benchmark_file = os.path.join('..', 'Input', 'piano_and_synth_arp_chord_mono.wav')

    def test_2_sin_waves(self):
        sr = nussl.constants.DEFAULT_SAMPLE_RATE
        dur = 3  # seconds
        length = dur * sr
        sine_wave_1 = np.sin(np.linspace(0, 440 * 2 * np.pi, length))
        sine_wave_2 = np.sin(np.linspace(0, 440 * 5 * 2 * np.pi, length))
        signal = sine_wave_1 + sine_wave_2
        test_audio = nussl.AudioSignal()
        test_audio.load_audio_from_array(signal)

        # Set up NMMF MFCC
        nmf_mfcc = nussl.NMF_MFCC(test_audio, num_sources=2, num_templates=2, distance_measure="euclidean",
                                  num_iterations=5)

        # and run
        nmf_mfcc.run()
        sources = nmf_mfcc.make_audio_signals()

        for i, source in enumerate(sources):
            output_file_name = str(i) + '.wav'
            source.write_audio_to_file(output_file_name)

    def test_piano_and_synth(self):
        # Load input file
        input_file_name = os.path.join('..', 'input', 'piano_and_synth_arp_chord_mono.wav')
        signal = nussl.AudioSignal(path_to_input_file=input_file_name)

        # Set up NMMF MFCC
        nmf_mfcc = nussl.NMF_MFCC(signal, num_sources=2, num_templates=6, distance_measure="euclidean",
                                  num_iterations=150)
        # and run
        nmf_mfcc.run()
        sources = nmf_mfcc.make_audio_signals()
        for i, source in enumerate(sources):
            output_file_name = str(i) + '.wav'
            source.write_audio_to_file(output_file_name)

def freeze_nmf_mfcc_values():
    path = NMFMFCCUnitTests.path_to_benchmark_file

    signal = nussl.AudioSignal(path)
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

freeze_nmf_mfcc_values()