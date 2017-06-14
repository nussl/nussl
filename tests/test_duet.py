#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import unittest
import nussl
import numpy as np
import scipy.io.wavfile as wav
import os
import warnings
from nussl import Duet as dt


class DuetUnitTests(unittest.TestCase):
    input_file_name = os.path.join('..', 'Input', 'dev1_female3_inst_mix.wav')
    signal = nussl.AudioSignal(path_to_input_file=input_file_name)

    # def test_setup(self):
    #     input_file_name = os.path.join('..', 'Input', 'dev1_female3_inst_mix.wav')
    #     signal = nussl.AudioSignal(path_to_input_file=input_file_name)

    #     duet = nussl.Duet(signal, 3)
    #     duet.run()

    def test_refact_duet(self):
		refact_duet = dt.Duet(signal, num_sources=3)
		refact_duet_result = refact_duet.run()
		duet = nussl.Duet(signal, 3)
		nussel_duet_result = duet.run()
		assert refact_duet_result == nussel_duet_result

		

    # def test_setup_duet(self):


if __name__ == '__main__':
    unittest.main()

