#!/usr/bin/env python
# -*- coding: utf-8 -*-



import warnings

import numpy as np

from . import separation_base
from ..core import stft_utils
from ..core import constants

from .ft2d import FT2D
from .repet import Repet
from .repet_sim import RepetSim


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
    def __init__(self, input_audio_signal, separation_method, separation_args=None,
                 overlap_window_size=24, overlap_hop_size=12, overlap_window_type=constants.WINDOW_TRIANGULAR,
                 do_mono=False, use_librosa_stft=constants.USE_LIBROSA_STFT, stft_params=None):
        super(OverlapAdd, self).__init__(input_audio_signal=input_audio_signal)
        self.background = None
        self.foreground = None

        if stft_params is not None:
            self.stft_params.window_length = stft_params['window_length']
            self.stft_params.n_fft_bins = stft_params['window_length']
            self.stft_params.hop_length = stft_params['hop_length']

        self.use_librosa_stft = use_librosa_stft
        self.overlap_window_size = overlap_window_size
        self.overlap_hop_size = overlap_hop_size
        self.overlap_window_type = overlap_window_type

        self.window_samples = int(np.round(self.audio_signal.sample_rate * self.overlap_window_size))
        self.hop_samples = int(np.round(self.audio_signal.sample_rate * self.overlap_hop_size))
        self.overlap_samples = self.window_samples - self.hop_samples

        # Set the separation method
        self._separation_method = None
        self._separation_args = separation_args if separation_args else {}
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
        return list(OverlapAdd._valid_separation_methods.values())

    @staticmethod
    def valid_separation_method_names():
        """
        Returns a list of strings of class names that OverlapAdd can use as its separation method
        Returns:

        """
        return [method.__name__ for method in list(OverlapAdd._valid_separation_methods.values())]

    @staticmethod
    def _valid_methods_lower():
        """Case invariant (lowercase) version of self._valid_separation_methods dictionary.
        """
        return {OverlapAdd._format(name): obj for name, obj in list(OverlapAdd._valid_separation_methods.items())}

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
        error = ValueError(
            f"Invalid separation method for OverlapAdd! Got {value}, but valid methods"
            f"are: {', '.join(list(self._valid_separation_methods.keys()))}"
        )
        if value is None:
            raise error

        if isinstance(value, str):
            if self._format(value) in list(self._valid_methods_lower().keys()):
                # The user input a string with a valid method name. It should be in our dictionary
                self._separation_method = self._valid_methods_lower()[self._format(value)]
            else:
                # Oops. Can't find it in our dictionary
                raise error

        elif issubclass(value, separation_base.SeparationBase) and \
                        value in list(self._valid_separation_methods.values()):
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
        return str(list(filter(str.isalnum, string))).lower()

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

        self._separation_instance = self.separation_method(
            self.audio_signal, use_librosa_stft=self.use_librosa_stft, **self._separation_args
        )

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
        # if our window is larger than the total number of samples in the file,
        # just run the algorithm like normal
        if self.audio_signal.signal_length < self.window_samples + self.hop_samples:
            warnings.warn(
                'input_audio_signal length is less than one window.'
                f' Running {self.separation_method_name} normally...'
            )
            self._setup_instance()
            self._separation_instance.run()
            self.background, _ = self._separation_instance.make_audio_signals()
            return self.background

        nearest_hop = (int(self.audio_signal.signal_length / self.hop_samples) - 1) * self.hop_samples
        pad_amount = (nearest_hop + self.window_samples) - self.audio_signal.signal_length
        if self.audio_signal.signal_length % self.window_samples == 0:
            pad_amount = 0
        self.audio_signal.zero_pad(0, pad_amount)
        self._setup_instance()
        
        background_array = np.zeros_like(self.audio_signal.audio_data)
        #background_array = np.pad(background_array, ((0, 0), (pad_amount, pad_amount)), 'constant')
        

        # Make the window for multiple channels
        window = stft_utils.make_window(self.overlap_window_type, self.window_samples)
        window *= 2*(self.hop_samples/self.window_samples)
        window = np.vstack([window for _ in range(self.audio_signal.num_channels)])
        start = 0
        end = self.window_samples
        starts = np.arange(0, self.audio_signal.signal_length, self.hop_samples)
        starts = starts[:-1]

        # Main overlap-add loop
        for segment, start in enumerate(starts):
            end = start + self.window_samples
            end = min(end, self.audio_signal.signal_length)

            # middle cases are straight forward
            this_window = window.copy()
            
            if segment == 0:
                this_window[:, :self.hop_samples] = 1
            elif segment == len(starts) - 1:
                this_window[:, self.hop_samples:] = 1
                        
            
            unwindowed = self._set_active_region_and_run(start, end)
            windowed = np.multiply(unwindowed.audio_data, this_window[:, :unwindowed.signal_length])
            background_array[:, start:end] += windowed
        

        self.audio_signal.set_active_region_to_default()
        self.background = self.audio_signal.make_copy_with_audio_data(background_array, verbose=False)
        self.background.crop_signal(0, pad_amount)
        self.audio_signal.crop_signal(0, pad_amount)
        return self.background

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

        return [self.background, self.audio_signal - self.background]
