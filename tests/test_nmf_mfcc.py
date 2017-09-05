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

    def load_benchmarks(self):
        benchmark_dict = {}
        directory = 'nmf_mfcc_reference/nmf_mfcc_benchmarks'
        for filename in os.listdir(directory):
            key = os.path.splitext(filename)[0]
            file_path = os.path.join('nmf_mfcc_reference', 'nmf_mfcc_benchmarks', filename)
            value = np.load(file_path)
            benchmark_dict[key] = value
        return benchmark_dict


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

    def benchmark_nmf_mfcc(self):
        # Load input file
        input_file_name = os.path.join('..', 'input', 'piano_and_synth_arp_chord_mono.wav')
        signal = nussl.AudioSignal(path_to_input_file=input_file_name)

        # Set up NMMF MFCC
        nmf_mfcc = nussl.NMF_MFCC(signal, num_sources=2, num_templates=6, distance_measure="euclidean",
                                  num_iterations=150)
        # and run
        nmf_mfcc.run()

        benchmark_activation_matrix = np.load(os.path.join('nmf_mfcc_reference', 'nmf_mfcc_benchmarks', 'benchmark_activation_matrix.npy'))
        benchmark_templates_matrix = np.load(os.path.join('nmf_mfcc_reference', 'nmf_mfcc_benchmarks', 'benchmark_templates_matrix.npy'))
        benchmark_labeled_templates = np.load(os.path.join('nmf_mfcc_reference', 'nmf_mfcc_benchmarks', 'benchmark_labeled_templates.npy'))
        benchmark_masks = np.load(os.path.join('nmf_mfcc_reference', 'nmf_mfcc_benchmarks', 'benchmark_masks.npy'))

        nmf_mfcc.make_audio_signals()
        benchmark_sources = np.load(os.path.join('nmf_mfcc_reference', 'nmf_mfcc_benchmarks', 'benchmark_sources.npy'))

        assert np.all(benchmark_activation_matrix == nmf_mfcc.activation_matrix)
        assert np.all(benchmark_templates_matrix == nmf_mfcc.templates_matrix)
        assert np.all(benchmark_labeled_templates == nmf_mfcc.labeled_templates)
        assert np.all(benchmark_masks == nmf_mfcc.masks)
        assert np.all(benchmark_sources == nmf_mfcc.sources)

