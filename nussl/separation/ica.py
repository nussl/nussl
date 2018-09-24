#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sklearn

import separation_base
from ..core import constants
from ..core.audio_signal import AudioSignal
from ..core import utils


class ICA(separation_base.SeparationBase):
    """Separate sources using the Independent Component Analysis, given observations of the audio scene.
    nussl's ICA is a wrapper for sci-kit learn's implementation of FastICA, and provides a way to interop between
    nussl's :ref:`AudioSignal` objects and FastICA.

    References:
        `sci-kit learn FastICA <http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.fastica.html>`_

    Parameters:
        observations_list: multichannel AudioSignal object containing each observation of the mixture in each channel.
        Can prepare this input_audio_signal from multiple AudioSignal objects using
        ICA.numpy_observations_to_audio_signal(observations), where observations is a list of AudioSignal objects.

    """

    def __init__(self, observations_list=None, sample_rate=constants.DEFAULT_SAMPLE_RATE,
                 max_iterations=None, random_seed=None, fast_ica_kwargs=None):
        observations_signal = self._validate_observations_list(observations_list, sample_rate)
        super(ICA, self).__init__(input_audio_signal=observations_signal)

        # FastICA setup attributes
        self.num_components = self.audio_signal.num_channels

        self.fast_ica_kwargs = fast_ica_kwargs if isinstance(fast_ica_kwargs, dict) else {}
        self.max_iters = self._get_default_or_key(max_iterations, 'max_iter', self.fast_ica_kwargs)
        self.random_seed = self._get_default_or_key(random_seed, 'random_state', self.fast_ica_kwargs)

        # Results attributes
        self.estimated_sources = None
        self.estimated_mixing_params = None
        self.mean = None

    def _validate_observations_list(self, observations_list, sample_rate=None):
        """
        Validation for the observation list, can be a numpy array or list of AudioSignals with mono audio data
        Args:
            observations_list:

        Returns:

        """

        if isinstance(observations_list, np.ndarray):
            return self.numpy_observations_to_audio_signal(observations_list, sample_rate)

        elif isinstance(observations_list, list):
            return self.audio_signal_observations_to_audio_signal(observations_list)

        else:
            raise ValueError('Expected numpy array or list of AudioSignal objects,'
                             ' but got {}!'.format(type(observations_list)))

    @staticmethod
    def numpy_observations_to_audio_signal(observations, sample_rate=constants.DEFAULT_SAMPLE_RATE):
        """

        Args:
            observations (:obj:`np.ndarray`):
            sample_rate (int):

        Returns:

        """
        assert isinstance(observations, np.ndarray), 'Observations must be a numpy array!'
        if observations.ndim > 1\
                and observations.shape[constants.CHAN_INDEX] > observations.shape[constants.LEN_INDEX]:
            observations = observations.T

        return AudioSignal(audio_data_array=observations, sample_rate=sample_rate)

    @staticmethod
    def audio_signal_observations_to_audio_signal(observations):
        """

        Args:
            observations:

        Returns:

        """
        observations = utils.verify_audio_signal_list_strict(observations)

        if not all(observations[0].signal_length == o.signal_length for o in observations):
            raise ValueError('All observation AudioSignal objects must have the same length!')

        if not all(o.is_mono for o in observations):
            raise ValueError('All AudioSignals in observations_list must be mono!')

        observation_data = np.vstack([o.audio_data for o in observations])
        return AudioSignal(audio_data_array=observation_data, sample_rate=observations[0].sample_rate)

    @staticmethod
    def _get_default_or_key(default_value, key, dict_):
        if default_value is not None:
            return default_value
        elif key in dict_:
            return dict_[key]
        else:
            return None

    def run(self):
        """

        Returns:
            background (AudioSignal): An AudioSignal object with repeating background in background.audio_data
            (to get the corresponding non-repeating foreground run self.make_audio_signals())

        Example:
             ::

        """
        ica = sklearn.decomposition.FastICA(n_components=self.num_components, random_state=self.random_seed,
                                            max_iter=self.max_iters, **self.fast_ica_kwargs)

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
        self.estimated_sources = [AudioSignal(audio_data_array=ica_output[i, :],
                                              sample_rate=self.audio_signal.sample_rate)
                                  for i in range(ica_output.shape[0])]

        return self.estimated_sources

    def make_audio_signals(self):
        """ Returns the background and foreground audio signals. You must have run ICA.run() prior
        to calling this function. This function will return None if run() has not been called.

        Returns:
            Audio Signals (list): list

        EXAMPLE:
             ::
        """
        if self.estimated_sources is None:
            raise ValueError('ICA.run() must be run prior to calling ICA.make_audio_signals()!')

        return self.estimated_sources
