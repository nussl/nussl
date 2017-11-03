#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import nussl
import numpy as np
import scipy.io
import os

class TestRepet(unittest.TestCase):

    @staticmethod
    def _load_final_matlab_results():
        back_path = os.path.join('tests', 'repet_reference', 'repet_matlab_results', 'mix3_matlab_repet_background')
        fore_path = os.path.join('tests', 'repet_reference', 'repet_matlab_results', 'mix3_matlab_repet_foreground')
        matlab_background = scipy.io.loadmat(back_path)['back'].T
        matlab_foreground = scipy.io.loadmat(fore_path)['fore'].T

        return matlab_background, matlab_foreground

    def test_output(self):
        path = os.path.join('Input', 'mix3.wav')
        signal = nussl.AudioSignal(path)

        repet = nussl.Repet(signal, matlab_fidelity=True)
        background_mask = repet.run()

        background_sig, foreground_sig = repet.make_audio_signals()

        matlab_background, matlab_foreground = self._load_final_matlab_results()

        assert np.allclose(background_sig.audio_data, matlab_background)
        assert np.allclose(foreground_sig.audio_data, matlab_foreground)
        assert background_mask

        i = 0

    def test_masks(self):
        pass
