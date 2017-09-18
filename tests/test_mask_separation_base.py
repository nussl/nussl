#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test class for MaskSeparationBase

"""

import unittest
import numpy as np
import warnings

import nussl


class MaskSeparationBaseUnitTests(unittest.TestCase):

    def test_setup(self):
        """
        Test many different ways for setting up MaskSeparationBase class
        Returns: Test fails if any of the setup tests fail when they should succeed or vice versa.

        """
        sig = nussl.AudioSignal()

        # Invalid ways to set up
        with self.assertRaises(ValueError):
            mask_type = 'bin'
            _ = nussl.MaskSeparationBase(sig, mask_type)

        with self.assertRaises(ValueError):
            mask_type = 'asdf'
            _ = nussl.MaskSeparationBase(sig, mask_type)

        with self.assertRaises(ValueError):
            mask_type = nussl.AudioSignal
            _ = nussl.MaskSeparationBase(sig, mask_type)

        # all valid ways
        mask_type = 'BiNaRy'
        mask_separator = nussl.MaskSeparationBase(sig, mask_type)
        assert mask_separator.mask_type == nussl.MaskSeparationBase.BINARY_MASK

        mask_type = nussl.separation.BinaryMask
        mask_separator = nussl.MaskSeparationBase(sig, mask_type)
        assert mask_separator.mask_type == nussl.MaskSeparationBase.BINARY_MASK

        # with self.assertRaises(warnings.warn):
        mask_type = nussl.separation.BinaryMask(np.zeros(shape=(512, 1024)))
        mask_separator = nussl.MaskSeparationBase(sig, mask_type)
        assert mask_separator.mask_type == nussl.MaskSeparationBase.BINARY_MASK

        mask_type = 'SOFT'
        mask_separator = nussl.MaskSeparationBase(sig, mask_type)
        assert mask_separator.mask_type == nussl.MaskSeparationBase.SOFT_MASK

        mask_type = nussl.separation.SoftMask
        mask_separator = nussl.MaskSeparationBase(sig, mask_type)
        assert mask_separator.mask_type == nussl.MaskSeparationBase.SOFT_MASK

        # with self.assertRaises(warnings.warn):
        mask_type = nussl.separation.SoftMask(np.zeros(shape=(512, 1024)))
        mask_separator = nussl.MaskSeparationBase(sig, mask_type)
        assert mask_separator.mask_type == nussl.MaskSeparationBase.SOFT_MASK

    def test_not_implemented(self):
        """
        Tests to make sure that run(), plot(), and make_audio_signals() raise
        NotImplementedErrors because they shouldn't be accessible at this level.
        Returns: Test fails if any of these functions do no raise exceptions

        """
        sig = nussl.AudioSignal()
        mask_separator = nussl.MaskSeparationBase(sig, nussl.separation.BinaryMask)
        with self.assertRaises(NotImplementedError):
            mask_separator.run()

        with self.assertRaises(NotImplementedError):
            mask_separator.plot('')

        with self.assertRaises(NotImplementedError):
            mask_separator.make_audio_signals()

    def test_json(self):
        """
        Test MaskSeparationBase going to and from json
        Returns:

        """
        sig = nussl.AudioSignal()
        mask_separator = nussl.MaskSeparationBase(sig, nussl.separation.BinaryMask)

        mask_json = mask_separator.to_json()

        new_mask_separator = nussl.MaskSeparationBase.from_json(mask_json)

        assert new_mask_separator == mask_separator


