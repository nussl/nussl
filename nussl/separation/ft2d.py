#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import copy
from scipy.ndimage.filters import maximum_filter, minimum_filter, uniform_filter

from ..core.audio_signal import AudioSignal
from ..core import constants
from . import mask_separation_base
from . import masks


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
    def __init__(self, input_audio_signal, high_pass_cutoff=100.0, neighborhood_size=(1, 25),
                 do_mono=False, use_librosa_stft=constants.USE_LIBROSA_STFT,
                 mask_type=mask_separation_base.MaskSeparationBase.SOFT_MASK,
                 filter_approach='original'):
        super(FT2D, self).__init__(input_audio_signal=input_audio_signal, mask_type=mask_type)
        self.high_pass_cutoff = high_pass_cutoff
        self.background = None
        self.foreground = None
        self.use_librosa_stft = use_librosa_stft
        self.neighborhood_size = neighborhood_size
        self.result_masks = None

        self.stft = None
        allowed_filter_approaches = ['original', 'local_std']
        if filter_approach not in allowed_filter_approaches:
            raise ValueError(f'filter approach must be one of {allowed_filter_approaches}')

        self.filter_approach = filter_approach

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

        self._compute_spectrograms()

        # separate the mixture background by masking
        background_masks = []
        foreground_masks = []
        for ch in range(self.audio_signal.num_channels):
            background_mask, foreground_mask = self.compute_ft2d_mask(self.ft2d, ch)
            background_mask[0:self.high_pass_cutoff, :] = 1  # high-pass filter the foreground
            background_masks.append(background_mask)
            foreground_masks.append(foreground_mask)

        background_masks = np.array(background_masks).transpose((1, 2, 0)).astype('float')
        foreground_masks = np.array(foreground_masks).transpose((1, 2, 0)).astype('float')

        _masks = [background_masks, foreground_masks]
        self.result_masks = []

        for mask in _masks:
            mask = masks.SoftMask(mask)
            if self.mask_type == self.BINARY_MASK:
                mask = mask.mask_to_binary()
            self.result_masks.append(mask)

        return self.result_masks

    
    def _compute_spectrograms(self):
        self.stft = self.audio_signal.stft(overwrite=True, remove_reflection=True,
                                           use_librosa=self.use_librosa_stft)
        self.ft2d = np.stack([np.fft.fft2(np.abs(self.stft[:, :, i]))
                              for i in range(self.audio_signal.num_channels)], 
                              axis = -1)

    def compute_ft2d_mask(self, ft2d, ch):
        if self.filter_approach == 'original':
            bg_ft2d, fg_ft2d = self.filter_local_maxima(ft2d[:, :, ch])
        elif self.filter_approach == 'local_std':
            bg_ft2d, fg_ft2d = self.filter_local_maxima_with_std(ft2d[:, :, ch])
        
        _stft = np.abs(self.stft)[:, :, ch]
        bg_stft = np.abs(np.fft.ifft2(bg_ft2d)) / (_stft + 1e-7)
        fg_stft = np.abs(np.fft.ifft2(fg_ft2d)) / (_stft + 1e-7)
        
        bg_mask = bg_stft / (np.maximum(bg_stft, fg_stft) + 1e-7)
        fg_mask = fg_stft / (np.maximum(bg_stft, fg_stft) + 1e-7)
        fg_mask = fg_mask * (1 - bg_mask)
        bg_mask = bg_mask * (1 - fg_mask)
        return bg_mask, fg_mask

    def filter_local_maxima_with_std(self, ft2d):
        data = np.abs(np.fft.fftshift(ft2d))
        data /= np.max(data)

        data_max = maximum_filter(data, self.neighborhood_size)
        data_min = minimum_filter(data, self.neighborhood_size)
        data_mean = uniform_filter(data, self.neighborhood_size)
        data_mean_sq = uniform_filter(data ** 2, self.neighborhood_size)
        data_std = np.sqrt(data_mean_sq - data_mean ** 2)

        maxima = ((data_max - data_mean) / data_std)
        fraction_of_local_max = (data == data_max)
        maxima *= fraction_of_local_max
        maxima = maxima.astype(float)
        maxima /= np.max(maxima)

        maxima = np.maximum(maxima, np.fliplr(maxima), np.flipud(maxima))
        maxima = np.fft.ifftshift(maxima)
        
        background_ft2d = np.multiply(maxima, ft2d)
        foreground_ft2d = np.multiply(1 - maxima, ft2d)
        return background_ft2d, foreground_ft2d


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
        sources = []
        residual = copy.deepcopy(self.audio_signal)
        for mask in self.result_masks:
            source = self.audio_signal.apply_mask(mask)
            source.istft(
                overwrite=True,
                truncate_to_length=self.audio_signal.signal_length
            )
            sources.append(source)
            residual -= source
        return sources
