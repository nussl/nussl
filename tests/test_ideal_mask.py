#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import os
import numpy as np

import nussl

class IdealMaskTestCase(unittest.TestCase):

    def setUp(self):
        dur = 30
        offset = 60
        premade_mixture_path = os.path.join('..', 'input', 'mixture', 'mixture.wav')
        vocals_path = os.path.join('..', 'input', 'mixture', 'vocals.wav')
        drums_path = os.path.join('..', 'input', 'mixture', 'drums.wav')

        self.premade_mixture = nussl.AudioSignal(premade_mixture_path, duration=dur, offset=offset)
        self.vocals = nussl.AudioSignal(vocals_path, duration=dur, offset=offset)
        self.drums = nussl.AudioSignal(drums_path, duration=dur, offset=offset)
        self.new_mixture = self.vocals + self.drums

    def test_setup(self):
        pass

    def test_run_premade(self):
        """
        Testing the IdealMask class using a pre-made mixture. The pre-made mixture SoftMask and BinaryMask
        Returns:

        """
        mask_types = [nussl.SoftMask, nussl.BinaryMask]

        for mask_class in mask_types:
            ideal_mask = nussl.IdealMask(self.premade_mixture, sources_list=[self.vocals, self.drums],
                                         mask_type=mask_class)
            masks = ideal_mask.run()
            assert all([isinstance(m, mask_class) for m in masks])

            masked_sources = ideal_mask.make_audio_signals()
            residual = ideal_mask.residual
            assert isinstance(residual, nussl.AudioSignal)

            reconstructed = masked_sources[0].audio_data + masked_sources[1].audio_data + residual.audio_data
            assert np.allclose(reconstructed, self.premade_mixture.audio_data)

    def test_run_new_mixture(self):
        mask_types = [nussl.SoftMask, nussl.BinaryMask]

        for mask_class in mask_types:
            ideal_mask = nussl.IdealMask(self.new_mixture, sources_list=[self.vocals, self.drums],
                                         mask_type=mask_class)
            masks = ideal_mask.run()
            assert all([isinstance(m, mask_class) for m in masks])
            ac = np.allclose(masks[0].mask + masks[1].mask, np.ones(masks[0].shape))

            masked_sources = ideal_mask.make_audio_signals()
            residual = ideal_mask.residual
            assert isinstance(residual, nussl.AudioSignal)

            reconstructed = masked_sources[0].audio_data + masked_sources[1].audio_data
            # assert np.allclose(reconstructed, self.premade_mixture.audio_data)






