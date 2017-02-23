#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nussl
import unittest
import numpy as np


class TestOverlapAdd(unittest.TestCase):

    def setUp(self):
        """
        Set up variables used in the tests.
        """
        self.valid_methods = [nussl.Repet, nussl.RepetSim, nussl.FT2D]
        self.invalid_methods = [nussl.Duet, nussl.OverlapAdd, nussl.NMF, nussl.SeparationBase, nussl.RPCA,
                           nussl.StftParams, nussl.AudioSignal, nussl.DistanceType, int, str, unittest.TestCase,
                           None]
        self.valid_method_names = [m.__name__ for m in self.valid_methods]
        self.invalid_method_names = [m.__name__ for m in self.invalid_methods if m is not None]
        self.invalid_method_names.append('None')

        num_samples = nussl.DEFAULT_SAMPLE_RATE * 300  # 300 seconds = 5 min
        sine_wave = np.sin(np.linspace(0, 100 * 2 * np.pi, num_samples))  # Freq = 100 Hz
        self.signal = nussl.AudioSignal(audio_data_array=sine_wave)
        self.signal.path_to_input_file = 'check/out/this/cool/path.wav'

    def test_overlap_add_setup(self):
        """
        Test setting up the OverlapAdd class in various different ways.
        """
        # Test valid methods
        for method in self.valid_methods:
            ola1 = nussl.OverlapAdd(self.signal, method)
            ola2 = nussl.OverlapAdd(self.signal, method.__name__)
            assert ola1.separation_method == ola2.separation_method == method

            instance = method(self.signal)
            assert type(ola1.separation_instance) == type(ola2.separation_instance) == type(instance)

        # These special cases should work too
        ola = nussl.OverlapAdd(self.signal, 'r-e-p*-e-t!@#')
        assert ola.separation_method == nussl.Repet

        ola = nussl.OverlapAdd(self.signal, 'repet_sim')
        assert ola.separation_method == nussl.RepetSim

        # 'repet_sim' is misspelled --> raise error
        with self.assertRaises(ValueError):
            ola = nussl.OverlapAdd(self.signal, 'repe_sim')
            assert ola.separation_method == nussl.RepetSim

        # Test invalid methods
        for method in self.invalid_methods:
            with self.assertRaises(ValueError):
                ola = nussl.OverlapAdd(self.signal, method)

        # Test invalid method names
        for method in self.invalid_method_names:
            with self.assertRaises(ValueError):
                ola = nussl.OverlapAdd(self.signal, method)

        # Test that variables are stored correctly
        params = nussl.StftParams(nussl.DEFAULT_SAMPLE_RATE)
        params.window_length = 4096
        params.hop_length = params.window_length // 4
        self.signal.stft_params = params
        for method in self.valid_methods:
            ola = nussl.OverlapAdd(self.signal, method, use_librosa_stft=False)
            assert ola.separation_instance.stft_params == params
            assert ola.separation_instance.audio_signal == self.signal
            assert ola.separation_instance.audio_signal.path_to_input_file == self.signal.path_to_input_file

    def test_overlap_add_simple(self):
        """
        Tests running the OverlapAdd method with default settings for everything. Does not check output.

        """
        for i, method in enumerate(self.valid_methods):
            ola = nussl.OverlapAdd(self.signal, method)
            ola.run()
            bk, fg = ola.make_audio_signals()

    def test_overlap_add_properties(self):
        """
        Tests to make sure the properties are set correctly.
        """
        assert sorted(nussl.OverlapAdd.valid_separation_methods()) == sorted(self.valid_methods)
        assert sorted(nussl.OverlapAdd.valid_separation_method_names()) == sorted(self.valid_method_names)

        for i, method in enumerate(self.valid_methods):
            ola = nussl.OverlapAdd(self.signal, method)
            assert ola.separation_method == method
            assert ola.separation_method_name == self.valid_method_names[i]
            assert type(ola.separation_instance) == method

        for i, method in enumerate(self.invalid_methods):
            ola = nussl.OverlapAdd(self.signal, nussl.Repet)
            with self.assertRaises(ValueError):
                ola.separation_method = method
            with self.assertRaises(ValueError):
                ola.separation_method = self.invalid_method_names[i]
