#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for Mask class
"""

from __future__ import division
import unittest
import numpy as np

import nussl


class MaskUnitTests(unittest.TestCase):

    def test_mask_setup(self):

        # make sure Mask does NOT work
        with self.assertRaises(NotImplementedError):
            arr = np.random.randint(0, 2, size=[6, 6])
            _ = nussl.MaskBase(arr)

    l = 1024
    h = 512
    ch = 1

    def test_soft_mask_setup(self):

        # SoftMask initialized with ints => bad
        with self.assertRaises(ValueError):
            arr = np.random.randint(0, 2, size=[6, 6])
            _ = nussl.SoftMask(arr)

        # SoftMask initialized out of range => bad
        with self.assertRaises(ValueError):
            arr = (np.random.random((6, 6)) * 10) - 5
            _ = nussl.SoftMask(arr)

        # not enough dimensions
        with self.assertRaises(ValueError):
            arr = np.random.random(10)
            _ = nussl.SoftMask(arr)

        # too many dimensions
        with self.assertRaises(ValueError):
            arr = np.random.random((10, 10, 10, 10))
            _ = nussl.SoftMask(arr)

        # SoftMask initialized correctly
        arr = np.random.random((self.h, self.l))
        soft_mask = nussl.SoftMask(arr)
        assert soft_mask.dtype.kind in np.typecodes['AllFloat']
        assert soft_mask.length == self.l
        assert soft_mask.height == self.h
        assert soft_mask.num_channels == self.ch
        assert np.all(soft_mask.get_channel(0) == arr)
        assert soft_mask.get_channel(0).shape == arr.shape

        with self.assertRaises(ValueError):
            soft_mask.get_channel(1)

        assert np.all(soft_mask.mask == np.expand_dims(arr, axis=nussl.STFT_CHAN_INDEX))
        assert soft_mask.mask.ndim == arr.ndim + 1

    def test_binary_mask_setup(self):

        # Try to initialize with floats far from 0.0 and 1.0 => bad
        with self.assertRaises(ValueError):
            arr = np.random.random((self.h, self.l))
            _ = nussl.BinaryMask(arr)

        # try to initialize with ints that are not 0, 1 => bad
        with self.assertRaises(ValueError):
            arr = np.random.randint(-10, 10, size=(self.h, self.l))
            _ = nussl.BinaryMask(arr)

        # try with floats that are very close to 0, 1, but above tolerance
        with self.assertRaises(ValueError):
            arr = np.random.randint(0, 2, size=(self.h, self.l)).astype('float')
            arr += np.random.random((self.h, self.l)) * 0.1
            _ = nussl.BinaryMask(arr)

        arr = np.random.random((self.h, self.l))
        soft_mask = nussl.SoftMask(arr)

        # Make a binary mask from SoftMax
        binary_mask = soft_mask.mask_to_binary()
        assert np.all([binary_mask.get_channel(0) == (arr > 0.5)])

    def test_chan(self):
        m = np.random.randint(0, 2, size=(1024, 512))
