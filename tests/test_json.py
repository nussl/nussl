#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import nussl
import numpy as np
import os


class TestJson(unittest.TestCase):

    def test_audio_signal(self):
        path = os.path.join('..', 'Input', 'mix1.wav')
        a = nussl.AudioSignal(path)
        a.stft()

        j = a.to_json()
        b = nussl.AudioSignal.from_json(j)
        worked = a == b
        return worked

    def test_stft_params(self):
        s = nussl.stft_utils.StftParams(44100)
        j = s.to_json()
        b = nussl.stft_utils.StftParams.from_json(j)
        worked = s == b
        return worked

    def test_repet(self):
        path = os.path.join('..', 'Input', 'mix1.wav')
        a = nussl.AudioSignal(path)
        r = nussl.Repet(a)
        r()

        j = r.to_json()
        f = nussl.Repet.from_json(j)
        worked = r == f
        return worked

    def test_duet(self):
        path = os.path.join('..', 'Input', 'dev1_female3_inst_mix.wav')
        a = nussl.AudioSignal(path)
        d = nussl.Duet(a, 3)
        d()

        j = d.to_json()
        e = nussl.Duet.from_json(j)
        worked = d == e
        return worked

    def test_repet_sim(self):
        path = os.path.join('..', 'Input', 'mix1.wav')
        a = nussl.AudioSignal(path)
        r = nussl.RepetSim(a)
        r()

        j = r.to_json()
        f = nussl.RepetSim.from_json(j)
        worked = r == f
        return worked

    @unittest.skip('')
    def test_nmf_mfcc(self):
        path = os.path.join('..', 'Input', 'piano_and_synth_arp_chord_mono.wav')
        a = nussl.AudioSignal(path)
        n = nussl.NMF_MFCC(a, num_sources=2)
        n()

        j = n.to_json()
        f = nussl.NMF_MFCC.from_json(j)
        worked = n == f
        return worked
