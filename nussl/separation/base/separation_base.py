#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import inspect
import json
import warnings

import numpy as np

from ...core import utils
from ...core import constants
from ... import AudioSignal

class SeparationBase(object):
    """Base class for all separation algorithms in nussl.

    Do not call this. It will not do anything.

    Parameters:
        input_audio_signal (AudioSignal). AudioSignal` object.
                            This will always be a copy of the provided AudioSignal object.
    """

    def __init__(self, input_audio_signal):
        if not isinstance(input_audio_signal, AudioSignal):
            raise ValueError('input_audio_signal is not an AudioSignal object!')

        self._audio_signal = None
        self.audio_signal = input_audio_signal

        if not self.audio_signal.has_data:
            warnings.warn('input_audio_signal has no data!')

            # initialize to empty arrays so that we don't crash randomly
            self.audio_signal.audio_data = np.array([])
            self.audio_signal.stft_data = np.array([[]])

    @property
    def sample_rate(self):
        """(int): Sample rate of :attr:`audio_signal`.
        Literally :attr:`audio_signal.sample_rate`.
        """
        return self.audio_signal.sample_rate

    @property
    def stft_params(self):
        """STFTParams object containing
        """
        return self.audio_signal.stft_params

    @property
    def audio_signal(self):
        """Copy of AudioSignal that is made on initialization.
        """
        return self._audio_signal

    @audio_signal.setter
    def audio_signal(self, input_audio_signal):
        self._audio_signal = copy.copy(input_audio_signal)

    def plot(self, **kwargs):
        """Plots relevant data for separation algorithm
        """
        pass

    def run(self):
        """Runs separation algorithm

        Raises:
            NotImplementedError: Cannot call base class
        """
        raise NotImplementedError('Cannot call base class.')

    def make_audio_signals(self):
        """Makes :class:`audio_signal.AudioSignal` objects after separation algorithm is run

        Raises:
            NotImplementedError: Cannot call base class
        """
        raise NotImplementedError('Cannot call base class.')

    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)
        return self.make_audio_signals()

    def __repr__(self):
        return f"{self.__class__.__name__} on {str(self.audio_signal)}"

    def __eq__(self, other):
        for k, v in list(self.__dict__.items()):
            if isinstance(v, np.ndarray):
                if not np.array_equal(v, other.__dict__[k]):
                    return False
            elif v != other.__dict__[k]:
                return False
        return True

    def __ne__(self, other):
        return not self == other

class SeparationException(Exception):
    pass