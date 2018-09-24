#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest
import json

import numpy as np
import nussl


class NMFMFCCUnitTests(unittest.TestCase):

    def setUp(self):
        # If our working directory is not the top level dir
        if os.path.basename(os.path.normpath(os.getcwd())) == 'tests':
            os.chdir('..')  # then go up one level

        input_mono = os.path.join('input', 'piano_and_synth_arp_chord_mono.wav')
        self.signal_mono = nussl.AudioSignal(input_mono)

        input_stereo = os.path.join('input', 'piano_and_synth_arp_chord_stereo.wav')
        self.signal_stereo = nussl.AudioSignal(input_stereo)

        self.n_src = 2  # number of sources in both of these files

    @staticmethod
    def load_benchmarks(self):
        benchmark_dict = {}
        directory = os.path.join('nmf_mfcc_reference', 'nmf_mfcc_benchmark_files')
        for filename in os.listdir(directory):
            key = os.path.splitext(filename)[0]
            file_path = os.path.join(directory, filename)
            value = np.load(file_path)
            benchmark_dict[key] = value
        return benchmark_dict

    def test_nmf_mfcc_initialization(self):
        # Test initializing the MFCC range

        # Set up the max of the MFCC range by only using an int
        nmf_mfcc = nussl.NMF_MFCC(self.signal_mono, num_sources=self.n_src,
                                  num_templates=6, num_iterations=5,
                                  mfcc_range=5)
        assert nmf_mfcc.mfcc_start == 1
        assert nmf_mfcc.mfcc_end == 5

        # Set up the MFCC range by using a list [min, max]
        nmf_mfcc = nussl.NMF_MFCC(self.signal_mono, num_sources=self.n_src,
                                  num_templates=6, num_iterations=5,
                                  mfcc_range=[3, 15])
        assert nmf_mfcc.mfcc_start == 3
        assert nmf_mfcc.mfcc_end == 15

        # Set up the MFCC range by using a tuple (min, max)
        nmf_mfcc = nussl.NMF_MFCC(self.signal_mono, num_sources=self.n_src,
                                  num_templates=6, num_iterations=5,
                                  mfcc_range=(2, 14))
        assert nmf_mfcc.mfcc_start == 2
        assert nmf_mfcc.mfcc_end == 14

    def test_random_seed_initialization(self):

        # Different random_seed and random_state, check if set separately
        kmean_kwargs = {'random_state': 0}
        nmf_mfcc = nussl.NMF_MFCC(self.signal_mono, num_sources=self.n_src,
                                  num_templates=6, num_iterations=5,
                                  random_seed=1, kmeans_kwargs=kmean_kwargs)
        assert nmf_mfcc.clusterer.random_state == 0
        assert nmf_mfcc.random_seed == 1

        # No random_seed, but random_state initialized
        kmean_kwargs = {'random_state': 0}
        nmf_mfcc = nussl.NMF_MFCC(self.signal_mono, num_sources=2,
                                  num_templates=6, num_iterations=5,
                                  kmeans_kwargs=kmean_kwargs)
        assert nmf_mfcc.clusterer.random_state == 0
        assert nmf_mfcc.random_seed is None

        # No random_state, but random_seed initialized
        nmf_mfcc = nussl.NMF_MFCC(self.signal_mono, num_sources=self.n_src, num_templates=6,
                                  num_iterations=5, random_seed=0)
        assert nmf_mfcc.clusterer.random_state == nmf_mfcc.random_seed == 0

    def test_2_sin_waves(self):
        sr = nussl.DEFAULT_SAMPLE_RATE
        dur = 3  # seconds
        length = dur * sr
        sine_wave_1 = np.sin(np.linspace(0, 440 * 2 * np.pi, length))
        sine_wave_2 = np.sin(np.linspace(0, 440 * 5 * 2 * np.pi, length))
        signal = sine_wave_1 + sine_wave_2
        test_audio = nussl.AudioSignal()
        test_audio.load_audio_from_array(signal)

        # Set up NMMF MFCC
        nmf_mfcc = nussl.NMF_MFCC(test_audio, num_sources=self.n_src,
                                  num_templates=2, num_iterations=15)

        # and run
        nmf_mfcc.run()
        nmf_mfcc.make_audio_signals()

    def test_piano_and_synth(self):
        # Set up NMMF MFCC
        nmf_mfcc = nussl.NMF_MFCC(self.signal_mono, num_sources=self.n_src * 2,
                                  num_templates=6, num_iterations=10,
                                  random_seed=0)

        # and run
        computed_masks = nmf_mfcc.run()
        estimated_sources = nmf_mfcc.make_audio_signals()
        assert len(estimated_sources) == self.n_src

        for source in estimated_sources:
            assert source.is_mono

        assert self.signal_mono == estimated_sources[0] + estimated_sources[1]

    def test_piano_and_synth_stereo(self):
        # Set up NMMF MFCC
        nmf_mfcc = nussl.NMF_MFCC(self.signal_stereo, num_sources=self.n_src,
                                  num_templates=6, num_iterations=15,
                                  random_seed=0)

        # and run
        nmf_mfcc.run()
        estimated_sources = nmf_mfcc.make_audio_signals()
        assert len(estimated_sources) == self.n_src

        for source in estimated_sources:
            assert source.is_stereo

    def test_piano_and_synth_stereo_to_mono(self):
        # Set up NMMF MFCC and convert input audio signal to mono
        nmf_mfcc = nussl.NMF_MFCC(self.signal_stereo, num_sources=self.n_src,
                                  num_templates=6, num_iterations=15,
                                  random_seed=0, distance_measure='euclidean',
                                  to_mono=True)

        assert nmf_mfcc.audio_signal.is_mono

        # and run
        nmf_mfcc.run()
        estimated_sources = nmf_mfcc.make_audio_signals()
        assert len(estimated_sources) == self.n_src

        for source in estimated_sources:
            assert source.is_mono

    def test_benchmark_nmf_mfcc(self):

        metadata_file = os.path.join('tests', 'nmf_mfcc_reference', 'nmf_mfcc_benchmark_files',
                                     'nmf_mfcc_benchmark_metadata.json')
        metadata = json.load(open(metadata_file, 'r'))
        print('Reading benchmark files from {}, '
              'version {}, made by {}'.format(metadata['save_time'],
                                              metadata['nussl_version'],
                                              metadata['made_by']))

        # Load the audio
        signal = nussl.AudioSignal(metadata['input_file'])

        # Set up NMMF MFCC
        params = metadata['params']  # Load params from metadata
        nmf_mfcc = nussl.NMF_MFCC(signal, **params)

        # run
        nmf_mfcc.run()

        benchmark_labeled_templates = np.load(os.path.join('tests', 'nmf_mfcc_reference',
                                                           'nmf_mfcc_benchmark_files',
                                                           'benchmark_labeled_templates.npy'))
        benchmark_masks = np.load(os.path.join('tests', 'nmf_mfcc_reference',
                                               'nmf_mfcc_benchmark_files',
                                               'benchmark_masks.npy'))

        nmf_mfcc.make_audio_signals()

        # Set the paths to empty so we can compare
        for source in nmf_mfcc.sources:
            source.path_to_input_file = ''
        benchmark_sources = np.load(os.path.join('tests', 'nmf_mfcc_reference',
                                                 'nmf_mfcc_benchmark_files',
                                                 'benchmark_sources.npy'))

        assert np.all(benchmark_sources == nmf_mfcc.sources)
        assert np.all(benchmark_labeled_templates == nmf_mfcc.labeled_templates)
        assert np.all(benchmark_masks == nmf_mfcc.result_masks)
