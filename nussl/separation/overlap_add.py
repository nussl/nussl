#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import warnings

import numpy as np

import separation_base
from ..core import stft_utils
from ..core import constants

from ft2d import FT2D
from repet import Repet
from repet_sim import RepetSim


class OverlapAdd(separation_base.SeparationBase):
    """
    Implements foreground/background separation using overlap/add with an arbitrary foreground/background
        separation scheme in nussl.

    Notes:
        Currently supports ``Repet``, ``RepetSim``, and ``FT2D``.

    Parameters:
        input_audio_signal (:class:`audio_signal.AudioSignal`): The :class:`audio_signal.AudioSignal` object that the 
        OverlapAdd algorithm will be run on. This makes a copy of ``input_audio_signal``

        separation_method:
        overlap_window_size:
        overlap_hop_size:
        overlap_window_type:
        do_mono:
        use_librosa_stft:

    Example:

    .. code-block:: python
        :linenos:
        
         import nussl
         
         signal = nussl.AudioSignal('path/to/audio.wav')
         
         ola = nussl.OverlapAdd(signal, nussl.Repet)  # initialize with class
         ola = nussl.OverlapAdd(signal, 'repet')  # initialize with string (case insensitive)
         ola.run()
         
    """
    def __init__(self, input_audio_signal, separation_method,
                 overlap_window_size=24, overlap_hop_size=12, overlap_window_type=constants.WINDOW_TRIANGULAR,
                 do_mono=False, use_librosa_stft=constants.USE_LIBROSA_STFT):
        super(OverlapAdd, self).__init__(input_audio_signal=input_audio_signal)
        self.background = None
        self.foreground = None

        self.use_librosa_stft = use_librosa_stft
        self.overlap_window_size = overlap_window_size
        self.overlap_hop_size = overlap_hop_size
        self.overlap_window_type = overlap_window_type

        self.window_samples = int(np.round(self.audio_signal.sample_rate * self.overlap_window_size))
        self.hop_samples = int(np.round(self.audio_signal.sample_rate * self.overlap_hop_size))
        self.overlap_samples = self.window_samples - self.hop_samples

        # Set the separation method
        self._separation_method = None
        self._separation_instance = None

        self.separation_method = separation_method

        if do_mono:
            self.audio_signal.to_mono(overwrite=True)

    # Here is where we store valid separation methods for OverlapAdd.
    # To add a new method for OverlapAdd, follow these four steps:
    #
    # 0) import the class at the top of this file.
    # 1) Add the class object here (not an instance), as below:
    repet = Repet
    repet_sim = RepetSim
    ft2d = FT2D

    # 2) And populate this dictionary with your new method like so:
    _valid_separation_methods = {Repet.__name__: repet,
                                 RepetSim.__name__: repet_sim,
                                 FT2D.__name__: ft2d}

    # 3) Add it to the tests in tests/test_overlap_add.py. If you don't, I will hunt you down!!!

    @staticmethod
    def valid_separation_methods():
        """
        Returns a list of class objects that OverlapAdd can use as its separation method
        Returns:

        """
        return OverlapAdd._valid_separation_methods.values()

    @staticmethod
    def valid_separation_method_names():
        """
        Returns a list of strings of class names that OverlapAdd can use as its separation method
        Returns:

        """
        return [method.__name__ for method in OverlapAdd._valid_separation_methods.values()]

    @staticmethod
    def _valid_methods_lower():
        """Case invariant (lowercase) version of self._valid_separation_methods dictionary.
        """
        return {OverlapAdd._format(name): obj for name, obj in OverlapAdd._valid_separation_methods.items()}

    @property
    def separation_method_name(self):
        """
        Returns the name of the current separation object
        Returns:

        """
        return str(self.separation_instance)

    @property
    def separation_method(self):
        """
        Returns the current separation object
        Returns:

        """
        return self._separation_method

    @separation_method.setter
    def separation_method(self, value):
        """
        Sets self.separation_method to value
        Resets self._separation_instance to None
        Args:
            value:

        Returns:

        """
        error = ValueError("Invalid separation method for OverlapAdd! \n" +
                           "Got {0}, but valid methods are: {1}"
                           .format(value, ', '.join(self._valid_separation_methods.keys())))
        if value is None:
            raise error

        if isinstance(value, str):
            if self._format(value) in self._valid_methods_lower().keys():
                # The user input a string with a valid method name. It should be in our dictionary
                self._separation_method = self._valid_methods_lower()[self._format(value)]
            else:
                # Oops. Can't find it in our dictionary
                raise error

        elif issubclass(value, separation_base.SeparationBase) and \
                        value in self._valid_separation_methods.values():
            # The user gave us a class, so we use that
            self._separation_method = value

        else:
            raise error

        self._setup_instance()

    @staticmethod
    def _format(string):
        """ Formats a class name correctly for self._valid_methods_lower.
            Strips all non-alphanumeric chars and makes lowercase.
        """
        return str(filter(str.isalnum, string)).lower()

    @property
    def separation_instance(self):
        """ This the actual instance of the separation algorithm. If you need to make any modifications to
        the default behavior of the algorithm, this is where you would do it.

        Returns:
            (:obj:`SeparationBase`) instance of the separation method.
        Examples:
            ::


        """
        return self._separation_instance

    def _setup_instance(self):
        # instantiate the separation method here
        if self.separation_method is None:
            raise Exception('Cannot separate before separation_method is set!')

        self._separation_instance = self.separation_method(self.audio_signal, use_librosa_stft=self.use_librosa_stft)

    def __str__(self):
        name = super(OverlapAdd, self).__str__()
        name += ':' + self.separation_method.__name__ if self.separation_method is not None else ''
        return name

    def run(self):
        """

        Returns:
            background (:obj:`AudioSignal`): An AudioSignal object with background in background.audio_data
            (to get the corresponding non-repeating foreground run self.make_audio_signals())

        Example:
             ::

        """
        if self._separation_instance is None:
            self._setup_instance()

        # if our window is larger than the total number of samples in the file,
        # just run the algorithm like normal
        if self.audio_signal.signal_length < self.window_samples + self.hop_samples:
            warnings.warn('input_audio_signal length is less than one window. '
                          'Running {} normally...'.format(self.separation_method_name))

            self._separation_instance.run()
            self.background, _ = self._separation_instance.make_audio_signals()
            return self.background

        background_array = np.zeros_like(self.audio_signal.audio_data)

        # Make the window for multiple channels
        window = stft_utils.make_window(self.overlap_window_type, 2 * self.overlap_samples)
        window = np.vstack([window for _ in range(self.audio_signal.num_channels)])

        # Main overlap-add loop
        for start, end in self._next_window():

            if start == 0:
                # First window is a partial window
                first_window = window[:, -self.hop_samples:]
                unwindowed = self._set_active_region_and_run(start, self.hop_samples)
                background_array[:, :self.hop_samples] = np.multiply(unwindowed.audio_data, first_window)

            elif end >= self.audio_signal.signal_length:
                # Last window is a partial window
                remaining = self.audio_signal.signal_length - start
                last_window = window[:, remaining:] if remaining != window.shape[-1] else window
                last_window[:, self.overlap_samples:] = 1  # only do part of the window

                unwindowed = self._set_active_region_and_run(start, self.audio_signal.signal_length)
                background_array[:, start:] += np.multiply(unwindowed.audio_data, last_window)

            else:
                # middle cases are straight forward
                unwindowed = self._set_active_region_and_run(start, end)
                background_array[:, start:end] += np.multiply(unwindowed.audio_data, window)

        self.audio_signal.set_active_region_to_default()
        self.background = self.audio_signal.make_copy_with_audio_data(background_array, verbose=False)
        return self.background

    def _next_window(self):
        """
        Generator that calculates the start and end sample indices for the next window.
        Yields:

        """
        n_segments = 1 + int((self.audio_signal.signal_length-self.window_samples) / self.hop_samples)

        # We went to return values larger than the signal length so we know when the last
        # n_segments = 1 + int( self.audio_signal.signal_length / self.hop_samples)

        for segment in range(n_segments):
            start = segment * self.hop_samples
            end = start + self.window_samples

            yield start, end

    def _set_active_region_and_run(self, start, end):
        self._separation_instance.audio_signal.set_active_region(start, end)
        self._separation_instance.run()
        bkgnd, _ = self._separation_instance.make_audio_signals()
        return bkgnd


    def make_audio_signals(self):
        """ Returns the background and foreground audio signals. You must have run :func:`run()` prior
        to calling this function. This function will raise ValueError if :func:`run()` has not been called.

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
