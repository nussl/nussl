#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import nussl
import os

class NMFMFCCUnitTests(unittest.TestCase):

    # Update this if the benchmark file changes and rerun freeze_duet_values() (below)
    path_to_benchmark_file = os.path.join('..', 'Input', 'piano_and_synth_arp_chord_mono.wav')

    def load_benchmarks(self):
        benchmark_dict = {}
        directory = 'nmf_mfcc_reference/nmf_mfcc_benchmarks'
        for filename in os.listdir(directory):
            key = os.path.splitext(filename)[0]
            file_path = os.path.join('nmf_mfcc_reference', 'nmf_mfcc_benchmarks', filename)
            value = np.load(file_path)
            benchmark_dict[key] = value
        return benchmark_dict

    def test_nmf_mfcc_initialization(self):
        # Test initializing the MFCC range

        # Load input file
        input_file_name = os.path.join('..', 'input', 'piano_and_synth_arp_chord_mono.wav')
        signal = nussl.AudioSignal(path_to_input_file=input_file_name)

        # Set up the max of the MFCC range by only using an int
        nmf_mfcc = nussl.NMF_MFCC(signal, num_sources=2, num_templates=6, mfcc_range=5, num_iterations=5)
        assert nmf_mfcc.mfcc_start == 1
        assert nmf_mfcc.mfcc_end == 5

        nmf_mfcc.run()
        sources = nmf_mfcc.make_audio_signals()

        # Set up the MFCC range by using a list [min, max]
        nmf_mfcc = nussl.NMF_MFCC(signal, num_sources=2, num_templates=6, mfcc_range=[3, 15], num_iterations=5)
        assert nmf_mfcc.mfcc_start == 3
        assert nmf_mfcc.mfcc_end == 15

        # Set up the MFCC range by using a tuple (min, max)
        nmf_mfcc = nussl.NMF_MFCC(signal, num_sources=2, num_templates=6, mfcc_range=(2, 14), num_iterations=5)
        assert nmf_mfcc.mfcc_start == 2
        assert nmf_mfcc.mfcc_end == 14

    def test_random_seed_initialization(self):
        # Load input file
        input_file_name = os.path.join('..', 'input', 'piano_and_synth_arp_chord_mono.wav')
        signal = nussl.AudioSignal(path_to_input_file=input_file_name)

        # Differing random_seed and random_state, check if set separately
        kmean_kwargs = {'random_state': 0}
        nmf_mfcc = nussl.NMF_MFCC(signal, num_sources=2, num_templates=6, num_iterations=5, random_seed=1,
                                kmeans_kwargs=kmean_kwargs)
        assert nmf_mfcc.clusterer.random_state == 0
        assert nmf_mfcc.random_seed == 1

        # No random_seed, but random_state initialized
        kmean_kwargs = {'random_state': 0}
        nmf_mfcc = nussl.NMF_MFCC(signal, num_sources=2, num_templates=6, num_iterations=5, kmeans_kwargs=kmean_kwargs)
        assert nmf_mfcc.clusterer.random_state == 0
        assert nmf_mfcc.random_seed == None

        # No random_state, but random_seed initialized
        nmf_mfcc = nussl.NMF_MFCC(signal, num_sources=2, num_templates=6, num_iterations=5, random_seed=0)
        assert nmf_mfcc.clusterer.random_state == nmf_mfcc.random_seed == 0

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
        nmf_mfcc = nussl.NMF_MFCC(test_audio, num_sources=2, num_templates=2, num_iterations=15)

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
        nmf_mfcc = nussl.NMF_MFCC(signal, num_sources=2, num_templates=6, num_iterations=50, random_seed=0)
        # and run
        nmf_mfcc.run()
        sources = nmf_mfcc.make_audio_signals()

    def test_piano_and_synth_stereo(self):
        # Load input file
        input_file_name = os.path.join('..', 'input', 'piano_and_synth_arp_chord_stereo.wav')
        signal = nussl.AudioSignal(path_to_input_file=input_file_name)

        # Set up NMMF MFCC
        nmf_mfcc = nussl.NMF_MFCC(signal, num_sources=2, num_templates=6, num_iterations=150, random_seed=0)
        # and run
        nmf_mfcc.run()
        sources = nmf_mfcc.make_audio_signals()

    def test_piano_and_synth_stereo_to_mono(self):
        # Load input file
        input_file_name = os.path.join('..', 'input', 'piano_and_synth_arp_chord_stereo.wav')
        signal = nussl.AudioSignal(path_to_input_file=input_file_name)

        # Set up NMMF MFCC and convert input audio signal to mono
        nmf_mfcc = nussl.NMF_MFCC(signal, num_sources=2, num_templates=6, distance_measure="euclidean",
                                  num_iterations=15, random_seed=0, convert_to_mono=True)
        # and run
        nmf_mfcc.run()
        sources = nmf_mfcc.make_audio_signals()

    def benchmark_nmf_mfcc(self):
        # Load input file
        input_file_name = os.path.join('..', 'input', 'piano_and_synth_arp_chord_mono.wav')
        signal = nussl.AudioSignal(path_to_input_file=input_file_name)

        # Set up NMMF MFCC
        nmf_mfcc = nussl.NMF_MFCC(signal, num_sources=2, num_templates=6, distance_measure="euclidean",
                                  num_iterations=100, random_seed=0)
        # and run
        nmf_mfcc.run()

        benchmark_labeled_templates = np.load(os.path.join('nmf_mfcc_reference', 'nmf_mfcc_benchmarks',
                                                           'benchmark_labeled_templates.npy'))
        benchmark_masks = np.load(os.path.join('nmf_mfcc_reference', 'nmf_mfcc_benchmarks', 'benchmark_masks.npy'))

        nmf_mfcc.make_audio_signals()
        benchmark_sources = np.load(os.path.join('nmf_mfcc_reference', 'nmf_mfcc_benchmarks', 'benchmark_sources.npy'))

        assert np.all(benchmark_labeled_templates == nmf_mfcc.labeled_templates)
        assert np.all(benchmark_masks == nmf_mfcc.masks)
        for i in range(nmf_mfcc.num_sources):
            assert np.all(benchmark_sources[i].audio_data == nmf_mfcc.sources[i].audio_data)
        assert all(benchmark_sources[i] == nmf_mfcc.sources[i] for i in range(len(nmf_mfcc.sources)))
