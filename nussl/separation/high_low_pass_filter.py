#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implements a simple high/low pass filter for audio source separation
"""

from __future__ import division
import numpy as np
import scipy.signal

import mask_separation_base
from ideal_mask import IdealMask


class HighLowPassFilter(mask_separation_base.MaskSeparationBase):
    """
    A simple high/low pass filter that creates a mask in the time frequency representation
    """

    def __init__(self, input_audio_signal, high_pass_cutoff_hz, do_fir_filter=False,
                 force_recompute_stft=False,
                 mask_type=mask_separation_base.MaskSeparationBase.BINARY_MASK):
        super(HighLowPassFilter, self).__init__(input_audio_signal=input_audio_signal,
                                                mask_type=mask_type)
        self.high_pass_cutoff_hz = high_pass_cutoff_hz

        self.should_do_fir_filter = do_fir_filter
        self.do_stft = force_recompute_stft
        self.stft = None

        self.high_pass_mask = None
        self.low_pass_mask = None

        self.high_pass_signal = None
        self.low_pass_signal = None

    def run(self):
        """

        Returns:

        """

        if self.should_do_fir_filter:

            # This implementation is based off of the sci-py cookbook:
            # https://scipy-cookbook.readthedocs.io/items/FIRFilter.html

            # Set up filter parameters
            nyquist_rate = self.audio_signal.sample_rate / 2
            width = 5.0 / nyquist_rate  # Hz
            ripple_db = 60

            # Convert cutoff frequency
            cutoff_hz = self.high_pass_cutoff_hz / nyquist_rate  # Hz

            # Make the filter
            filter_order, beta = scipy.signal.kaiserord(ripple_db, width)
            filter_coeffs = scipy.signal.firwin(filter_order, cutoff_hz, window=('kaiser', beta))

            # Apply the filter to every channel in the mixture
            low_pass_array = []
            for ch in self.audio_signal.get_channels():
                low_pass_array.append(scipy.signal.lfilter(filter_coeffs, [1.0], ch))

            # Make a new AudioSignal object with filtered signal
            low_pass_array = np.array(low_pass_array)
            self.low_pass_signal = self.audio_signal.make_copy_with_audio_data(low_pass_array,
                                                                               verbose=False)
            self.high_pass_signal = self.audio_signal - self.low_pass_signal

            # Make masks
            ideal_mask = IdealMask(self.audio_signal, [self.high_pass_signal, self.low_pass_signal],
                                   mask_type=self.mask_type)
            self.high_pass_mask, self.low_pass_mask = ideal_mask.run()

        else:
            # This is the more simple of the two filtering methods. Here we just zero out STFT bins

            # Compute the spectrogram and find the closest frequency bin to the cutoff freq
            self._get_stft()
            closest_freq_bin = self.audio_signal.get_closest_frequency_bin(self.high_pass_cutoff_hz)

            # Make masks
            self.low_pass_mask = self.ones_mask(self.stft.shape)
            self.low_pass_mask.mask[closest_freq_bin:, :, :] = 0

            self.high_pass_mask = self.low_pass_mask.invert_mask()

        self.result_masks = [self.low_pass_mask, self.high_pass_mask]

        return self.result_masks

    def _get_stft(self):
        """
        Computes the spectrogram of the input audio signal
        Returns:

        """
        if not self.audio_signal.has_stft_data or self.do_stft:
            self.stft = self.audio_signal.stft(overwrite=True)
        else:
            self.stft = self.audio_signal.stft_data

    def make_audio_signals(self):
        """

        Returns:

        """
        if self.low_pass_mask is None or self.high_pass_mask is None:
            raise ValueError('Must call run() before calling make_audio_signals()!')

        if not self.should_do_fir_filter:
            self.high_pass_signal = self.audio_signal.apply_mask(self.high_pass_mask)
            self.high_pass_signal.istft(overwrite=True,
                                        truncate_to_length=self.audio_signal.signal_length)

            self.low_pass_signal = self.audio_signal.apply_mask(self.low_pass_mask)
            self.low_pass_signal.istft(overwrite=True,
                                       truncate_to_length=self.audio_signal.signal_length)

        return self.low_pass_signal, self.high_pass_signal
