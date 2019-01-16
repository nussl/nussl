#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import os

import nussl


class RunAndEvalUnitTest(unittest.TestCase):

    def test_simple(self):
        drums_path = os.path.join('Input', 'src1.wav')
        flute_path = os.path.join('Input', 'src2.wav')

        drums = nussl.AudioSignal(drums_path)
        flute = nussl.AudioSignal(flute_path)
        flute.truncate_samples(drums.signal_length)

        gains = [1.0, 0.75, 0.5, 0.25, 0.0]  # gain settings
        drum_sigs = [drums.make_copy_with_audio_data(drums.audio_data * g) for g in gains]  # drums with different gains
        mixtures = [d + flute for d in drum_sigs]  # mix everything together
        true_sources = [[flute, d] for d in drum_sigs]

        repet_sim = nussl.RepetSim
        repet_kwargs = {}

        scores_sim = nussl.run_and_eval_prf(repet_sim, repet_kwargs, mixtures, true_sources)

        repet = nussl.Repet
        scores_repet = nussl.run_and_eval_prf(repet, repet_kwargs, mixtures, true_sources)

        i = 0
