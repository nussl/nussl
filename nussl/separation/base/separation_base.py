import copy
import warnings

import numpy as np

from ... import AudioSignal, play_utils


class SeparationBase(object):
    """Base class for all separation algorithms in nussl.

    Do not call this. It will not do anything.

    Parameters:
        input_audio_signal (AudioSignal). AudioSignal` object.
            This will always be a copy of the provided AudioSignal object.
    """

    def __init__(self, input_audio_signal):
        self.metadata = {}
        self._audio_signal = None
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

    def interact(self, add_residual=False, source='upload', label=None, share=False):
        """
        Uses gradio to create a small interactive interface
        for the separation algorithm. Fair warning, there
        may be some race conditions with this...

        When you call this from a notebook, the interface will be displayed
        below the cell. When you call this from a regular Python script, you'll see a 
        link print out (a localhost link and a gradio link if you
        called this with sharing on). The sessions will last for the duration
        of the notebook or the script.

        To use this functionality, you must install gradio: `pip install gradio`.

        Args:
            add_residual: Whether or not to add the residual signal.
            source: Either "upload" (upload a file to separate), or "microphone", record.
            share: Whether or not to create a public gradio link.
            kwargs: Keyword arguments to gradio.

        Example:
        
            >>> import nussl
            >>> nussl.separation.primitive.HPSS(
            >>>     nussl.AudioSignal()).interact()

        """
        try:
            import gradio
        except: # pragma: no cover
            raise ImportError(
                "To use this functionality, you must install gradio: "
                "pip install gradio.")
                
        def _separate(file_obj): # pragma: no cover
            mix = AudioSignal(file_obj.name)
            self.audio_signal = mix
            estimates = self()
            if add_residual:
                estimates.append(mix - estimates[0])

            estimates = {f'Estimate {i}': s for i, s in enumerate(estimates)}
            html = play_utils.multitrack(estimates, ext='.mp3', display=False)
            
            return html

        if label is None: label = f"Separation via {type(self).__name__}"

        audio_in = gradio.inputs.Audio(source=source, type="file", label=label)

        gradio.Interface(
            fn=_separate, 
            inputs=audio_in, 
            outputs="html",
        ).launch(share=share)

    def run(self, *args, audio_signal=None, **kwargs):
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

    def get_metadata(self, to_str=False, **kwargs):
        """
        Returns metadata associated with this separation algorithm.

        Args:
            to_str (bool): Whether to return the metadata as a string.

        Returns:
            Formatted metadata if `to_str` is True, else metadata dict.

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
