#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import unittest
import nussl
import numpy as np
import audio_signal

import scipy.io.wavfile as wav
import os
import warnings

import sys
sys.path.insert(1, 'C:\\Users\\danielfelixkim\\Documents\\GitHub\\InteractiveAudioLab\\nussl\\nussl')
import Duet as dt

class DuetUnitTests(unittest.TestCase):
    # Load input file
    input_file_name = os.path.join('..', 'Input', 'dev1_female3_inst_mix.wav')
    signal = audio_signal.AudioSignal(path_to_input_file=input_file_name)


    # def test_setup(self):
    #     input_file_name = os.path.join('..', 'Input', 'dev1_female3_inst_mix.wav')
    #     signal = nussl.AudioSignal(path_to_input_file=input_file_name)

    #     duet = nussl.Duet(signal, 3)
    #     duet.run()

  #   def test_refact_duet(self):

		# refact_duet = dt.Duet(self.signal, num_sources=3)
		# refact_duet_result = refact_duet.run()
		# nussel_duet = nussl.Duet(self.signal, num_sources=3)
		# nussel_duet_result = nussel_duet.run()
		# assert refact_duet_result == nussel_duet_result

    # def test_make_histogram(self):

    # def test_find_peaks2(self):

    # def test_convert_peaks(self):

    # def test_compute_masks(self):

    # def test_convert_time_domain(self):

    def test_setup_duet_1_channel(self):
        #Test with one channel, should throw value error
        pass
    

if __name__ == '__main__':
    unittest.main()

