#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage.filters import maximum_filter, minimum_filter

from ..core.audio_signal import AudioSignal
from ..core import constants
import mask_separation_base
import masks


class FT2D(mask_separation_base.MaskSeparationBase):
    """Implements foreground/background separation using the 2D Fourier Transform

    Parameters:
        input_audio_signal: (AudioSignal object) The AudioSignal object that has the
                            audio data that REPET will be run on.
        high_pass_cutoff: (Optional) (float) value (in Hz) for the high pass cutoff filter.
        do_mono: (Optional) (bool) Flattens AudioSignal to mono before running the algorithm
            (does not effect the input AudioSignal object)
        use_librosa_stft: (Optional) (bool) Calls librosa's stft function instead of nussl's

    """
    def __init__(self, input_audio_signal, high_pass_cutoff=None, neighborhood_size=(1, 25),
                 do_mono=False, use_librosa_stft=constants.USE_LIBROSA_STFT,
                 mask_type=mask_separation_base.MaskSeparationBase.SOFT_MASK):
        super(FT2D, self).__init__(input_audio_signal=input_audio_signal, mask_type=mask_type)
        self.high_pass_cutoff = 100.0 if high_pass_cutoff is None else float(high_pass_cutoff)
        self.background = None
        self.foreground = None
        self.use_librosa_stft = use_librosa_stft
        self.neighborhood_size = neighborhood_size
        self.result_masks = None

        self.stft = None
        self.ft2d = None

        if do_mono:
            self.audio_signal.to_mono(overwrite=True)

    def run(self):
        """

        Returns:
            background (AudioSignal): An AudioSignal object with repeating background in
            background.audio_data
            (to get the corresponding non-repeating foreground run self.make_audio_signals())

        Example:
             ::

        """
        # High pass filter cutoff freq. (in # of freq. bins), +1 to match MATLAB implementation
        self.high_pass_cutoff = int(np.ceil(self.high_pass_cutoff * (self.stft_params.n_fft_bins - 1) /
                                            self.audio_signal.sample_rate)) + 1

        # the MATLAB implementation had
        self._compute_spectrograms()

        # separate the mixture background by masking
        background_stft = []
        background_mask = []
        for i in range(self.audio_signal.num_channels):
            repeating_mask = self.compute_ft2d_mask(self.ft2d[:, :, i])
            repeating_mask[0:self.high_pass_cutoff, :] = 1  # high-pass filter the foreground
            background_mask.append(repeating_mask)

            # apply mask
            stft_with_mask = repeating_mask * self.stft[:, :, i]
            background_stft.append(stft_with_mask)

        background_stft = np.array(background_stft).transpose((1, 2, 0))
        self.background = AudioSignal(stft=background_stft,
                                      sample_rate=self.audio_signal.sample_rate)
        self.background.istft(self.stft_params.window_length, self.stft_params.hop_length,
                              self.stft_params.window_type,
                              overwrite=True, use_librosa=self.use_librosa_stft,
                              truncate_to_length=self.audio_signal.signal_length)

        background_mask = np.array(background_mask).transpose((1, 2, 0)).astype('float')
        background_mask = masks.SoftMask(background_mask)
        if self.mask_type == self.BINARY_MASK:
            background_mask = background_mask.mask_to_binary(self.mask_threshold)

        self.result_masks = [background_mask, background_mask.inverse_mask()]

        return self.result_masks
    
    def _compute_spectrograms(self):
        self.stft = self.audio_signal.stft(overwrite=True, remove_reflection=True,
                                           use_librosa=self.use_librosa_stft)
        self.ft2d = np.stack([np.fft.fft2(np.abs(self.stft[:, :, i]))
                              for i in range(self.audio_signal.num_channels)], axis = -1)

    def compute_ft2d_mask(self, ft2d):
        bg_ft2d, fg_ft2d = self.filter_local_maxima(ft2d)
        bg_stft = np.fft.ifft2(bg_ft2d)
        fg_stft = np.fft.ifft2(fg_ft2d)
        bg_mask = bg_stft > fg_stft
        return bg_mask

    def filter_local_maxima(self, ft2d):
        data = np.abs(np.fft.fftshift(ft2d))
        data /= np.max(data)
        threshold = np.std(data)
        
        data_max = maximum_filter(data, self.neighborhood_size)
        maxima = (data == data_max)
        data_min = minimum_filter(data, self.neighborhood_size)
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0
        maxima = np.maximum(maxima, np.fliplr(maxima), np.flipud(maxima))
        maxima = np.fft.ifftshift(maxima)
        
        background_ft2d = np.multiply(maxima, ft2d)
        foreground_ft2d = np.multiply(1 - maxima, ft2d)
        return background_ft2d, foreground_ft2d

    def make_audio_signals(self):
        """ Returns the background and foreground audio signals. You must have run FT2D.run() prior
        to calling this function. This function will return None if run() has not been called.

        Returns:
            Audio Signals (List): 2 element list.

                * bkgd: Audio signal with the calculated background track
                * fkgd: Audio signal with the calculated foreground track

        EXAMPLE:
             ::
        """
        if self.background is None:
            raise ValueError('Cannot make audio signals prior to running algorithm!')

        foreground_array = self.audio_signal.audio_data - self.background.audio_data
        self.foreground = self.audio_signal.make_copy_with_audio_data(foreground_array)
        return [self.background, self.foreground]
