#!/usr/bin/env python
# -*- coding: utf-8 -*-

import WindowAttributes
import Constants
import AudioSignal


class SeparationBase(object):
    """Base class for all separation algorithms.

    Do not call this. It will not do anything.

    Parameters:
        window_attributes (WindowAttributes): WindowAttributes for the separation algorithm. Defaults to
         WindowAttributes.WindowAttributes(self.sample_rate)
        sample_rate (Optional[int]): Sample rate. Defaults to Constants.DEFAULT_SAMPLE_RATE
        audio_signal (Optional[np.array]): Audio signal in array form. Defaults to AudioSignal.AudioSignal()

    """

    def __init__(self, window_attributes=None, sample_rate=None, audio_signal=None):

        if sample_rate is not None:
            self.sample_rate = sample_rate
        else:
            self.sample_rate = Constants.DEFAULT_SAMPLE_RATE

        if window_attributes is not None:
            self.window_attributes = window_attributes
        else:
            self.window_attributes = WindowAttributes.WindowAttributes(self.sample_rate)

        if audio_signal is not None:
            self.audio_signal = audio_signal
        else:
            self.audio_signal = AudioSignal.AudioSignal()

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
