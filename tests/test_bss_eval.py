#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for BSS Eval classes
"""

import unittest
import copy

import numpy as np

import nussl


class BSSEvalUnitTests(unittest.TestCase):

    def setUp(self):
        sample_rate = nussl.DEFAULT_SAMPLE_RATE
        signal_duration = 60  # seconds
        num_samples = sample_rate * signal_duration
        time = np.linspace(0, signal_duration, num_samples)

        sine = np.sin(2 * time)
        square = np.sign(np.sin(3 * time))
        raw_signal = np.c_[sine, square]  # two channel 'mixture'
        self.signal = nussl.AudioSignal(audio_data_array=raw_signal, sample_rate=sample_rate)

    def test_bss_eval_sources_simple(self):

        # mixture is the same as the estimated sources
        bss = nussl.evaluation.BSSEvalSources(self.signal, self.signal)
        bss.evaluate()

        scores = bss.scores

    def test_bss_eval_sources_simple2(self):
        mult = [0.5, 0.75, 1.0]
        signal_list = [copy.copy(self.signal) * m for m in mult]
        est_list = [copy.copy(self.signal) * m for m in mult[::-1]]

        # mixture is in reverse order to the estimated sources
        bss = nussl.evaluation.BSSEvalSources(signal_list, est_list)
        bss.evaluate()

        scores = bss.scores

    def test_bss_eval_images_simple(self):

        # mixture is the same as the estimated sources
        bss = nussl.evaluation.BSSEvalImages(self.signal, self.signal)
        bss.evaluate()

        scores = bss.scores

    def test_bss_eval_images_simple2(self):
        mult = [0.5, 0.75, 1.0]
        signal_list = [copy.copy(self.signal) * m for m in mult]
        est_list = [copy.copy(self.signal) * m for m in mult[::-1]]

        # mixture is in reverse order to the estimated sources
        bss = nussl.evaluation.BSSEvalImages(signal_list, est_list)
        bss.evaluate()

        scores = bss.scores

    def test_bss_eval_simple2(self):
        mult = [0.5, 0.75, 1.0]
        self.signal.to_mono(overwrite=True)
        signal_list = [copy.copy(self.signal) * m for m in mult]
        est_list = [(copy.copy(self.signal) * m) for m in mult[::-1]]

        # mixture is in reverse order to the estimated sources
        # bss = nussl.evaluation.BSSEvalV4( signal_list, est_list)
        # bss.evaluate()
        #
        # scores = bss.scores

    def test_bss_eval_v4(self):
        mir1k_dir = '/Users/ethanmanilow/Documents/Github/SourceSeparationPyCharm/demo_files/MIR-1K'
        mix, vox, acc = next(nussl.datasets.mir1k(mir1k_dir))
        mix.to_mono(overwrite=True)

        r = nussl.RepetSim(mix)
        r()
        bg, fg = r.make_audio_signals()

        est_dict = {'vocals': fg, 'accompaniment': bg}
        gt_dict = {'vocals': vox, 'accompaniment': acc}
        bss = nussl.evaluation.BSSEvalV4(mix, gt_dict, est_dict)
        scores1 = bss.evaluate()

        bss = nussl.evaluation.BSSEvalV4(mix, [vox, acc], [fg, bg])
        scores2 = bss.evaluate()

        assert scores1 == scores2
