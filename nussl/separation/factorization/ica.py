import copy

import numpy as np
import sklearn

from .. import SeparationBase
from ... import AudioSignal
from ...core import utils


class ICA(SeparationBase):
    """
    Separate sources using the Independent Component Analysis, given 
    observations of the audio scene. nussl's ICA is a wrapper for sci-kit learn's 
    implementation of FastICA, and provides a way to interop between 
    nussl's :ref:`AudioSignal` objects and FastICA.

    References:
        `sci-kit learn FastICA <http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.fastica.html>`_

    Args:
        audio_signals: list of AudioSignal objects containing the observations 
          of the mixture. Will be converted into a single multichannel AudioSignal.
        
        max_iterations (int): Max number of iterations to run ICA for. Defaults to 200.

        **kwargs: Additional keyword arguments that will be passed to 
          `sklearn.decomposition.FastICA`
    """

    def __init__(self, audio_signals, max_iterations=200, **kwargs):
        super().__init__(input_audio_signal=audio_signals)

        # FastICA setup attributes
        self.num_components = self.audio_signal.num_channels
        self.kwargs = kwargs
        self.max_iterations = max_iterations

        # Results attributes
        self.estimated_sources = None
        self.estimated_mixing_params = None
        self.mean = None
        self.ica_output = None

    @property
    def audio_signal(self):
        """
        Copy of AudioSignal that is made on initialization.
        """
        return self._audio_signal

    @audio_signal.setter
    def audio_signal(self, audio_signals):
        """
        Takes a list of audio signals and constructs a single multichannel audio signal
        object.

        Args:
            audio_signal (list or AudioSignal): Either a multichannel audio signal,
              or a list of AudioSignals containing the observations.
        """

        if isinstance(audio_signals, list):
            audio_signals = utils.verify_audio_signal_list_strict(audio_signals)
            audio_data = np.vstack([s.audio_data for s in audio_signals])
            audio_signal = audio_signals[0].make_copy_with_audio_data(audio_data)
            self._audio_signal = audio_signal
        elif isinstance(audio_signals, AudioSignal):
            self._audio_signal = copy.deepcopy(audio_signals)

    def run(self):
        ica = sklearn.decomposition.FastICA(
            n_components=self.num_components, max_iter=self.max_iterations,
            **self.kwargs)

        # save for normalizing the estimated signals
        max_input_amplitude = np.max(np.abs(self.audio_signal.audio_data))

        # run ICA
        ica_output = ica.fit_transform(self.audio_signal.audio_data.T).T

        # now normalize the estimated signals
        max_output_amplitude = np.max(np.abs(ica_output))
        ica_output /= max_output_amplitude
        ica_output *= max_input_amplitude

        # store the resultant computations
        self.estimated_mixing_params = ica.mixing_
        self.mean = ica.mean_
        self.ica_output = ica_output

        return self.ica_output

    def make_audio_signals(self):
        estimated_sources = [
            AudioSignal(
                audio_data_array=self.ica_output[i, :],
                sample_rate=self.audio_signal.sample_rate)
            for i in range(self.ica_output.shape[0])
        ]
        return estimated_sources
