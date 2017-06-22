#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import warnings

import nussl.config
import nussl.utils
import nussl.spectral_utils
import mask_separation_base
import masks


class IdealMask(mask_separation_base.MaskSeparationBase):
    """Separate sources using the ideal binary or soft mask from ground truth

    Parameters:
        input_audio_signal: (AudioSignal object) The AudioSignal object that has the
                            audio data that REPET will be run on.
        high_pass_cutoff: (Optional) (float) value (in Hz) for the high pass cutoff filter.
        do_mono: (Optional) (bool) Flattens AudioSignal to mono before running the algorithm (does not effect the
                        input AudioSignal object)
        use_librosa_stft: (Optional) (bool) Calls librosa's stft function instead of nussl's

    """

    def __init__(self, input_audio_mixture, sources_list,
                 mask_type=mask_separation_base.MaskSeparationBase.SOFT_MASK,
                 use_librosa_stft=nussl.config.USE_LIBROSA_STFT):
        super(IdealMask, self).__init__(input_audio_signal=input_audio_mixture, mask_type=mask_type)

        self.sources = nussl.utils._verify_audio_signal_list_strict(sources_list)

        # Make sure input_audio_signal has the same settings as sources_list
        if self.audio_signal.sample_rate != self.sources[0].sample_rate:
            raise ValueError('input_audio_signal must have the same sample rate as entries of sources_list!')
        if self.audio_signal.num_channels != self.sources[0].num_channels:
            raise ValueError('input_audio_signal must have the same number of channels as entries of sources_list!')

        self.estimated_masks = None
        self.estimated_sources = None
        self.use_librosa_stft = use_librosa_stft

    def run(self):
        """

        Returns:
            self.estimated_masks (list): 

        Example:
             ::

        """
        self._compute_spectrograms()
        self.estimated_masks = []

        for source in self.sources:
            cur_mask = np.divide(source.magnitude_spectrogram_data, self.audio_signal.magnitude_spectrogram_data)
            cur_mask /= np.max(cur_mask)

            mask = masks.SoftMask(cur_mask)
            if self.mask_type == self.BINARY_MASK:
                mask = mask.mask_to_binary(self.mask_threshold)

            self.estimated_masks.append(mask)

        return self.estimated_masks

    @property
    def residual(self):
        """
        
        Returns:

        """
        if self.estimated_masks is None:
            raise ValueError('Cannot calculate residual prior to running algorithm!')

        if self.estimated_sources is None:
            warnings.warn('Need to run self.make_audio_signals prior to calculating residual...')
            self.make_audio_signals()

        residual = self.audio_signal
        for source in self.estimated_sources:
            residual = residual - source

        return residual

    def _compute_spectrograms(self):
        self.audio_signal.stft(overwrite=True, remove_reflection=True, use_librosa=self.use_librosa_stft)
        for source in self.sources:
            source.stft(overwrite=True, remove_reflection=True, use_librosa=self.use_librosa_stft)

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
        if self.estimated_masks is None or self.audio_signal.stft_data.size <= 0:
            raise ValueError('Cannot make audio signals prior to running algorithm!')

        self.estimated_sources = []

        for cur_mask in self.estimated_masks:
            estimated_stft = np.multiply(cur_mask.mask, self.audio_signal.stft_data)
            new_signal = self.audio_signal.make_copy_with_stft_data(estimated_stft, verbose=False)
            new_signal.istft(self.stft_params.window_length, self.stft_params.hop_length,
                             self.stft_params.window_type, overwrite=True,
                             use_librosa=self.use_librosa_stft,
                             truncate_to_length=self.audio_signal.signal_length)

            self.estimated_sources.append(new_signal)

        return self.estimated_sources
