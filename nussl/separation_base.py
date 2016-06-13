#!/usr/bin/env python
# -*- coding: utf-8 -*-

import spectral_utils
import constants
import audio_signal


class SeparationBase(object):
    """Base class for all separation algorithms.

    Do not call this. It will not do anything.

    self.audio_signal, self.sample_rate, and self.stft_params are all properties that can have their setters and getters
    overridden in subclasses.

    Parameters:
        audio_signal (Optional[np.array]): Audio signal in array form. Defaults to AudioSignal.AudioSignal()
        sample_rate (Optional[int]): Sample rate. Defaults to Constants.DEFAULT_SAMPLE_RATE
        stft_params (StftParams): STFT parameters for the separation algorithm. Defaults to
         spectral_utils.StftParams(Constants.DEFAULT_SAMPLE_RATE)
    """

    def __init__(self, input_audio_signal):

        if input_audio_signal is not None:
            self.audio_signal = input_audio_signal
        else:
            self.audio_signal = audio_signal.AudioSignal()

    @property
    def sample_rate(self):
        return self.audio_signal.sample_rate

    @property
    def stft_params(self):
        return self.audio_signal.stft_params

    def plot(self, output_name, **kwargs):
        """Plots relevant data for separation algorithm

        Raises:
            NotImplementedError: Cannot call base class
        """
        raise NotImplementedError('Cannot call base class.')

    def run(self):
        """run separation algorithm

        Raises:
            NotImplementedError: Cannot call base class
        """
        raise NotImplementedError('Cannot call base class.')

    def make_audio_signals(self):
        """Makes AudioSignal objects after separation algorithm is run

        Raises:
            NotImplementedError: Cannot call base class
        """
        raise NotImplementedError('Cannot call base class.')

    def __call__(self):
        self.run()
