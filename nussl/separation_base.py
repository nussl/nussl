#!/usr/bin/env python
# -*- coding: utf-8 -*-

import audio_signal
import copy


class SeparationBase(object):
    """Base class for all separation algorithms.

    Do not call this. It will not do anything.

    Parameters:
        input_audio_signal: AudioSignal object. Defaults to a new audio_signal.AudioSignal() object.
                            This will always make a copy of the provided AudioSignal object.
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

    @property
    def audio_signal(self):
        return self.audio_signal

    @audio_signal.setter
    def audio_signal(self, input_audio_signal):
        self.audio_signal = copy.copy(input_audio_signal)

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
