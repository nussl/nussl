#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import nussl
import numpy as np
import scipy.io
import os


class TestRepet(unittest.TestCase):

    def setUp(self):
        self.audio_path = nussl.efz_utils.download_audio_file('mix3.wav')
        self.bg_mat_file = nussl.efz_utils.download_benchmark_file(
            'mix3_matlab_repet_background_bRuDiWq.mat')  # name got screwed up on the server. NBD
        self.fg_mat_file = nussl.efz_utils.download_benchmark_file(
            'mix3_matlab_repet_foreground.mat')

    def tearDown(self):
        os.remove(self.audio_path)
        os.remove(self.bg_mat_file)
        os.remove(self.fg_mat_file)

    def _load_final_matlab_results(self):
        matlab_background = scipy.io.loadmat(self.bg_mat_file)['back'].T
        matlab_foreground = scipy.io.loadmat(self.fg_mat_file)['fore'].T

        return matlab_background, matlab_foreground

    def test_output(self):
        signal = nussl.AudioSignal(self.audio_path)

        repet = nussl.Repet(signal, matlab_fidelity=True)
        background_mask = repet.run()

        background_sig, foreground_sig = repet.make_audio_signals()

        matlab_background, matlab_foreground = self._load_final_matlab_results()

        assert np.allclose(background_sig.audio_data, matlab_background)
        assert np.allclose(foreground_sig.audio_data, matlab_foreground)
        assert background_mask

    def test_masks(self):
        pass
