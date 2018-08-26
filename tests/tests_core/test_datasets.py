#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for nussl data sets
"""

import sys
import os
import unittest


try:
    # import from an already installed version
    import nussl
except:

    # can't find an installed version, import from right next door...
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if not path in sys.path:
        sys.path.insert(1, path)

    import nussl


class DataSetsUnitTests(unittest.TestCase):

    def test_data_set_utils(self):

        iKala_path = '/Users/ethanmanilow/Documents/School/Research/Predicting SDR values/iKala'
        mir1k_path = '/Users/ethanmanilow/Documents/School/Research/Predicting SDR values/prediction/Repet/mir_1k/MIR-1K'
        musdb_path = '/Users/ethanmanilow/Downloads/musdb18'

        for _ in nussl.datasets.musdb18(musdb_path, check_hash=False):
            i = 0

        for m, s, a in nussl.datasets.iKala(iKala_path):
            pass

        for m, s, a in nussl.datasets.mir1k(mir1k_path):
            pass