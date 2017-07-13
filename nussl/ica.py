#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.decomposition import FastICA

from audio_signal import AudioSignal
from nussl.separation import separation_base


class ICA(separation_base.SeparationBase):
    """Separate sources using the Independent Component Analysis, given observations of the audio scene.

    Parameters:
        input_audio_signal: multichannel AudioSignal object containing each observation of the mixture in each channel.
        Can prepare this input_audio_signal from multiple AudioSignal objects using
        ICA.transform_observations_to_audio_signal(observations), where observations is a list of AudioSignal objects.

    """

    def __init__(self, input_audio_signal=None):
        super(ICA, self).__init__(input_audio_signal=input_audio_signal)
        self.sources = None
        self.mixing = None

    @staticmethod
    def transform_observations_to_audio_signal(observations):
        lengths = set([o.signal_length for o in observations])
        if len(lengths) > 1:
            raise ValueError("All observation AudioSignal objects must have the same length!")
        observation_data = [o.audio_data for o in observations]
        observation_data = np.vstack(observation_data)
        observations = AudioSignal(audio_data_array=observation_data, sample_rate = observations[0].sample_rate)
        return observations


    def run(self):
        """

        Returns:
            background (AudioSignal): An AudioSignal object with repeating background in background.audio_data
            (to get the corresponding non-repeating foreground run self.make_audio_signals())

        Example:
             ::

        """
        ica = FastICA(n_components=self.audio_signal.num_channels)
        max_input_amplitude = np.max(np.abs(self.audio_signal.audio_data))
        sources = ica.fit_transform(self.audio_signal.audio_data.T).T
        max_output_amplitude = np.max(np.abs(sources))
        sources /= max_output_amplitude
        sources *= max_input_amplitude

        self.mixing = ica.mixing_
        self.mean = ica.mean_
        self.sources = []
        for i in range(sources.shape[0]):
            self.sources.append(AudioSignal(audio_data_array=sources[i, :], sample_rate=self.audio_signal.sample_rate))
        return self.sources

    def make_audio_signals(self):
        """ Returns the background and foreground audio signals. You must have run ICA.run() prior
        to calling this function. This function will return None if run() has not been called.

        Returns:
            Audio Signals (List): n element of list, where n is the number of channels of the input mixture

        EXAMPLE:
             ::
        """
        if self.sources is None:
            return None

        return self.sources
