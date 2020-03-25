import copy
import warnings

import numpy as np

from ... import AudioSignal


class SeparationBase(object):
    """Base class for all separation algorithms in nussl.

    Do not call this. It will not do anything.

    Parameters:
        input_audio_signal (AudioSignal). AudioSignal` object.
                            This will always be a copy of the provided AudioSignal object.
    """

    def __init__(self, input_audio_signal):
        self.audio_signal = input_audio_signal

    @property
    def sample_rate(self):
        """
        (int): Sample rate of :attr:`audio_signal`.
        Literally :attr:`audio_signal.sample_rate`.
        """
        return self.audio_signal.sample_rate

    @property
    def stft_params(self):
        """
        STFTParams object containing the STFT parameters of the copied AudioSignal.
        """
        return self.audio_signal.stft_params

    @property
    def audio_signal(self):
        """
        Copy of AudioSignal that is made on initialization.
        """
        return self._audio_signal

    def _preprocess_audio_signal(self):
        """
        This function should be implemented by the subclass. It can do things like
        take the STFT of the audio signal, or resample it to a desired sample rate,
        build the input data for a deep model, etc. Here, it does nothing.
        """
        pass

    @audio_signal.setter
    def audio_signal(self, input_audio_signal):
        """
        When setting the AudioSignal object for a separation algorithm (which
        can happen on initialization or later one), it is copied on set so
        as to not alter the data within the original audio signal. If the
        AudioSignal object has data, then it the function `_preprocess_audio_signal`
        is run, which is implemented by the subclass. 
        
        Args:
            input_audio_signal ([type]): [description]
        """
        if not isinstance(input_audio_signal, AudioSignal):
            raise ValueError('input_audio_signal is not an AudioSignal object!')

        self._audio_signal = copy.deepcopy(input_audio_signal)

        if self.audio_signal is not None:
            if not self.audio_signal.has_data:
                warnings.warn('input_audio_signal has no data!')

                # initialize to empty arrays so that we don't crash randomly
                self.audio_signal.audio_data = np.array([])
                self.audio_signal.stft_data = np.array([[]])
            else:
                self._preprocess_audio_signal()

    def run(self):
        """
        Runs separation algorithm.

        Raises:
            NotImplementedError: Cannot call base class
        """
        raise NotImplementedError('Cannot call base class.')

    def make_audio_signals(self):
        """
        Makes :class:`audio_signal.AudioSignal` objects after separation algorithm is run

        Raises:
            NotImplementedError: Cannot call base class
        """
        raise NotImplementedError('Cannot call base class.')

    def __call__(self, *args, audio_signal=None, **kwargs):
        if audio_signal is not None:
            self.audio_signal = audio_signal
        
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
