#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for Mask class
"""

from __future__ import division
import unittest
import numpy as np
import librosa

import nussl


class MaskUnitTests(unittest.TestCase):

    def test_mask_setup(self):

        # make sure Mask does NOT work
        with self.assertRaises(NotImplementedError):
            arr = np.random.randint(0, 2, size=[6, 6])
            _ = nussl.separation.MaskBase(arr)

    l = 1024
    h = 512
    ch = 1

    def test_soft_mask_setup(self):

        # SoftMask initialized with ints => bad
        with self.assertRaises(ValueError):
            arr = np.random.randint(0, 2, size=[6, 6])
            _ = nussl.separation.SoftMask(arr)

        # SoftMask initialized out of range => bad
        # TODO: Determine correct behavior for this case
        # with self.assertRaises(ValueError):
        #     arr = (np.random.random((6, 6)) * 10) - 5
        #     _ = nussl.separation.SoftMask(arr)

        # not enough dimensions
        with self.assertRaises(ValueError):
            arr = np.random.random(10)
            _ = nussl.separation.SoftMask(arr)

        # too many dimensions
        with self.assertRaises(ValueError):
            arr = np.random.random((10, 10, 10, 10))
            _ = nussl.separation.SoftMask(arr)

        # SoftMask initialized correctly
        arr = np.random.random((self.h, self.l))
        soft_mask = nussl.separation.SoftMask(arr)
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
            _ = nussl.separation.BinaryMask(arr)

        # try to initialize with ints that are not 0, 1 => bad
        with self.assertRaises(ValueError):
            arr = np.random.randint(-10, 10, size=(self.h, self.l))
            _ = nussl.separation.BinaryMask(arr)

        # try with floats that are very close to 0, 1, but above tolerance
        with self.assertRaises(ValueError):
            arr = np.random.randint(0, 2, size=(self.h, self.l)).astype('float')
            arr += np.random.random((self.h, self.l)) * 0.1
            _ = nussl.separation.BinaryMask(arr)

        arr = np.random.random((self.h, self.l))
        soft_mask = nussl.separation.SoftMask(arr)

        # Make a binary mask from SoftMax
        binary_mask = soft_mask.mask_to_binary()
        assert np.all([binary_mask.get_channel(0) == (arr > 0.5)])

    def test_chan(self):
        m = np.random.randint(0, 2, size=(1024, 512))

    def test_ones_and_zeros(self):
        shapes = [(1, 10, 100), (2, 1024, 1920)]

        for shape in shapes:
            ones_mask = nussl.separation.SoftMask.ones(shape)
            assert np.all(ones_mask.mask == np.ones(shape).astype('float'))

            ones_mask = nussl.separation.BinaryMask.ones(shape)
            assert np.all(ones_mask.mask == np.ones(shape).astype('bool'))

            zeros_mask = nussl.separation.SoftMask.zeros(shape)
            assert np.all(zeros_mask.mask == np.zeros(shape).astype('float'))

            zeros_mask = nussl.separation.BinaryMask.zeros(shape)
            assert np.all(zeros_mask.mask == np.zeros(shape).astype('bool'))

    def test_invert_mask(self):
        shape = (1, 100, 100)
        ones = np.ones(shape)

        soft_mask = nussl.separation.SoftMask(ones)

        inverse_soft_mask = soft_mask.invert_mask()
        assert np.all(inverse_soft_mask.mask == np.zeros(shape))

        binary_mask = nussl.separation.BinaryMask(ones)

        inverse_binary_mask = binary_mask.invert_mask()
        assert np.all(inverse_binary_mask.mask == np.zeros(shape))

    def _make_test_signal(self):

        fundamental_freq = 100  # Hz
        num_harmonics = 5
        duration = 10
        sample_rate = nussl.DEFAULT_SAMPLE_RATE
        num_samples = sample_rate * duration
        time = np.linspace(0, duration, num_samples)
        signal_array = np.zeros_like(time)

        for i in np.arange(1, num_harmonics+1):
            signal_array += np.sin(fundamental_freq * i * time)

        return nussl.AudioSignal(audio_data_array=signal_array)

    def test_apply_mask(self):
        signal = self._make_test_signal()
        signal.stft()
        hplp = nussl.separation.HighLowPassFilter(signal, 150, mask_type='binary')

        lp_mask, hp_mask = hplp.run()

        lp_signal = signal.apply_mask(lp_mask, overwrite=False)
        lp_signal.istft()

        librosa_mask = librosa.util.softmask(lp_mask.get_channel(0),
                                             signal.get_magnitude_spectrogram_channel(0),
                                             power=np.inf)
        librosa_mask = nussl.separation.BinaryMask(librosa_mask)

        lib_signal = signal.apply_mask(librosa_mask, overwrite=False)
        lib_signal.istft()

        i = 0







