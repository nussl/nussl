#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import spectral_utils
import separation_base
import constants
from audio_signal import AudioSignal
from scipy.ndimage.filters import maximum_filter, minimum_filter


class IdealMask(separation_base.SeparationBase):
    """Separate sources using the ideal binary or soft mask from ground truth

    Parameters:
        input_audio_signal: (AudioSignal object) The AudioSignal object that has the
                            audio data that REPET will be run on.
        high_pass_cutoff: (Optional) (float) value (in Hz) for the high pass cutoff filter.
        do_mono: (Optional) (bool) Flattens AudioSignal to mono before running the algorithm (does not effect the
                        input AudioSignal object)
        use_librosa_stft: (Optional) (bool) Calls librosa's stft function instead of nussl's

    """

    def __init__(self, input_audio_signal, use_librosa_stft=constants.USE_LIBROSA_STFT, sources=None):
        super(IdealMask, self).__init__(input_audio_signal=input_audio_signal)
        if sources is None:
            raise Exception('Cannot run IdealMask if there are no sources to derive a mask from!')
        self.sources = sources
        self.estimated = None
        self.use_librosa_stft = use_librosa_stft
        self.stft = None
        self.magnitude_spectrogram = None

    def run(self):
        """

        Returns:
            background (AudioSignal): An AudioSignal object with repeating background in background.audio_data
            (to get the corresponding non-repeating foreground run self.make_audio_signals())

        Example:
             ::

        """

        self._compute_spectrum()
        self.estimated = []

        for source in self.sources:
            mask = np.divide(np.abs(source.stft), self.magnitude_spectrogram)
            estimated_stft = np.multiply(mask, self.stft)
            self.estimated.append(AudioSignal(stft = estimated_stft, sample_rate=self.audio_signal.sample_rate))
            self.estimated[-1].istft(self.stft_params.window_length, self.stft_params.hop_length,
                                  self.stft_params.window_type, overwrite=True,
                                  use_librosa=self.use_librosa_stft,
                                  truncate_to_length=self.audio_signal.signal_length)
        residual = self.audio_signal
        for source in self.estimated:
            residual = residual - source
        self.estimated.append(residual)
        return self.estimated

    def _compute_spectrum(self):
        self.stft = self.audio_signal.stft(overwrite=True, remove_reflection=True, use_librosa=self.use_librosa_stft)
        self.magnitude_spectrogram = np.abs(self.stft)
        for source in self.sources:
            source.stft = source.stft(overwrite=True, remove_reflection=True, use_librosa=self.use_librosa_stft)


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
        if self.estimated is None:
            return None

        return self.estimated
