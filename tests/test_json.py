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
        s = nussl.StftParams(44100)
        j = s.to_json()
        b = nussl.StftParams.from_json(j)
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
        t = nussl.Repet(a)
        r()

        j = r.to_json()
        f = nussl.RepetSim.from_json(j)
        worked = r == f
        return worked