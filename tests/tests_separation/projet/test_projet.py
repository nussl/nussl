#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import unittest

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)

import nussl

class ProjetUnitTests(unittest.TestCase):

    def __init__(self):
        pass

    def PremsTest(self):
        # TODO: Refactor this into an actual test
        mixture = nussl.AudioSignal('../input/panned_mixture_four_sources.wav')

        projet = nussl.Projet(mixture, verbose=True, num_iterations=50, num_sources=4)
        projet.run()
        sources = projet.make_audio_signals()

        for i,m in enumerate(sources):
            m.write_audio_to_file('../input/projet_%d.wav' % i)
