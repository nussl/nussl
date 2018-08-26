#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Why are these tests here? (Are the functional?)
What did you do?
"""

from __future__ import division

import unittest
import nussl
import numpy as np
import scipy.io.wavfile as wav
import scipy.io
import os
import warnings


@unittest.skip("Pretty sure these tests never worked -- Keeping for posterity")
class DuetUnitTests(unittest.TestCase):

    def test_duet_outputs(self):
        input_file_name = os.path.join('..', 'input', 'dev1_female3_inst_mix.wav')
        signal = nussl.AudioSignal(path_to_input_file=input_file_name)
        refact_duet = nussl.Duet(signal, num_sources=3)
        refact_duet_result = refact_duet.run()
        duet = nussl.Duet(signal, 3)
        nussl_duet_result = duet.run()
        assert refact_duet_result == nussl_duet_result

    @staticmethod
    def _load_matlab_results():
        f_mat_path = os.path.join('duet_reference', 'rickard_duet', 'fmat')
        tf1_path = os.path.join('duet_reference', 'rickard_duet', 'tf1')
        tf2_path = os.path.join('duet_reference', 'rickard_duet', 'tf2')

        fmat = scipy.io.loadmat(f_mat_path)['back'].T
        delay_mat = scipy.io.loadmat(delay_mat_path)['fore'].T
        return sym_atn_mat, delay_mat

    def test_compute_spectrogram_1_channel(self):
        # Test with one channel, should throw value error
        num_samples = 100  # 1 second
        np_sin = np.sin(np.linspace(0, 100 * 2 * np.pi, num_samples))  # Freq = 100 Hz
        signal = nussl.AudioSignal(audio_data_array=np_sin)
        with self.assertRaises(ValueError):
            duet = nussl.Duet(signal, 3)
            duet._compute_spectrogram(duet.sample_rate)

    def test_compute_spectrogram_wmat(self):
        # Load MATLAB values
        # f_mat_path = os.path.join('duet_reference', 'rickard_duet', 'fmat')
        tf1_path = os.path.join('duet_reference', 'rickard_duet', 'tf1')
        tf2_path = os.path.join('duet_reference', 'rickard_duet', 'tf2')

        # fmat = scipy.io.loadmat(f_mat_path)['fmat']
        tf1_mat = scipy.io.loadmat(tf1_path)['tf1']
        tf2_mat = scipy.io.loadmat(tf2_path)['tf2']

        path = os.path.join('..', 'Input', 'dev1_female3_inst_mix.wav')
        signal = nussl.AudioSignal(path)
        duet = nussl.Duet(signal, 3)
        duet.stft_params.window_length = 1024
        duet.stft_params.window_type = nussl.WINDOW_RECTANGULAR
        duet.stft_params.hop_length = duet.stft_params.window_length

        duet_sft0, duet_sft1, duet_wmat = duet._compute_spectrogram(duet.sample_rate)
        assert np.allclose(duet_sft0, tf1_mat, atol=1e-06)
        assert np.allclose(duet_sft1, tf2_mat, atol=1e-06)
        # assert np.allclose(duet_wmat, fmat)

        # def test_compute_atn_delay(self):
        #     sym_atn_mat_path = os.path.join('duet_reference', 'rickard_duet', 'alpha_duet')
        #     delay_mat_path = os.path.join('duet_reference', 'rickard_duet', 'delta_duet')
        #     sym_atn_mat = scipy.io.loadmat(sym_atn_mat_path)['sym'].T
        #     delay_mat = scipy.io.loadmat(delay_mat_path)['delay'].T

        # def test_make_histogram(self):

        # def test_convert_peaks

        # def test_compute_masks

        # def test_convert_time_domain

        # test_smooth_matrix


if __name__ == '__main__':
    unittest.main()

