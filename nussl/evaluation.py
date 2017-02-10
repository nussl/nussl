#!/usr/bin/env python
# -*- coding: utf-8 -*-

from audio_signal import AudioSignal
from mir_eval.separation import *
import numpy as np

class Evaluation(object):
    """Lets you load ground truth AudioSignals and estimated AudioSignals and compute separation
    evaluation criteria (SDR, SIR, SAR and delta SDR, delta SIR, delta SAR).

    Parameters:
        ground_truth: ground truth audio sources that make up the mixture. Consists of a list of 
        AudioSignal objects that sum up to the mixture. This must be provided.
        estimated_sources: Estimated audio sources. These sum up to the mixture and don't have to 
        be in the same order as ground_truth. This can be provided later and swapped dynamically
        to compare different audio source separation approaches.
        ground_truth_labels: Labels for the sources in ground truth. Used to interpret the 
        results later.
        sample_rate: Sample rate for all ground truth and estimated sources. Defaults to 
        the sample rate of the first AudioSignal in ground_truth.
        do_mono: whether to do evaluation using mono sources to multichannel sources.
        compute_permutation: True if you can't guarantee that ground_truth and estimated_sources
        are in the same order. False if you can. It'll be a bit faster if False. 
        Defaults to True.
        segment_size: when computing evaluation metrics, you can do them by segment instead 
        of for the whole track. Segment size defines how long each segment is. 
        Defaults to 30 seconds.
        hop_size: when computing evaluation metrics, you can do them by segment instead of 
        for the whole track. Hop size defines how much to hop between segments. 
        Defaults to 15 seconds.
    Examples:
  
    """
    def __init__(self, ground_truth, estimated_sources, ground_truth_labels=None,
                 do_mono=False, compute_permutation=True, hop_size=15, segment_size=30):
        # Do input checking
        ground_truth = self._verify_input_list(ground_truth, do_mono)
        estimated_sources = self._verify_input_list(estimated_sources, do_mono)

        if do_mono:
            num_channels = 1
            [g.to_mono(overwrite=True) for g in ground_truth]
            [e.to_mono(overwrite=True) for e in estimated_sources]
        else:
            num_channels = ground_truth[0].num_channels

        # Now that everything is as we expect it, we can set our attributes
        self.ground_truth = ground_truth
        self.num_channels = num_channels
        self.do_mono = do_mono

        if ground_truth_labels is None:
            self.ground_truth_labels = ['Source %d' % i for i in range(len(ground_truth))]

        self.ground_truth_labels = ground_truth_labels

        self.estimated_sources = estimated_sources
        self.sample_rate = ground_truth[0].sample_rate
        
        self.compute_permutation = compute_permutation

        self.segment_size = segment_size
        self.hop_size = hop_size

    @staticmethod
    def _verify_input_list(audio_signal_list, do_mono):
        if isinstance(audio_signal_list, AudioSignal):
            audio_signal_list = [audio_signal_list]
        elif isinstance(audio_signal_list, list):
            if not all(isinstance(g, AudioSignal) for g in audio_signal_list):
                raise ValueError('All objects in ground_truth and estimated_sources list must be' +
                                 ' AudioSignal objects!')
        else:
            raise ValueError('Evaluation expects ground_truth and estimated_sources to be a list of' +
                             ' AudioSignal objects!')

        if not all(audio_signal_list[0].sample_rate == g.sample_rate for g in audio_signal_list):
            #  TODO: is this a huge deal? Maybe just throw a warning if not a problem...
            raise ValueError('Not all AudioSignal objects have the same sample rate.')

        if not do_mono:
            if not all(audio_signal_list[0].num_channels == g.num_channels for g in audio_signal_list):
                raise ValueError('Not all AudioSignal objects have the number of channels.')

        return audio_signal_list
    
    def validate(self):
        """

        Returns:

        """
        estimated_lengths = [x.signal_length for x in self.estimated_sources]
        reference_lengths = [x.signal_length for x in self.ground_truth]

        if len(set(estimated_lengths)) > 1:
            raise Exception('All AudioSignals in estimated_sources must be the same length!')
        if len(set(reference_lengths)) > 1:
            raise Exception('All AudioSignals in ground_truth must be the same length!')
    
    def to_mono(self):
        """

        Returns:

        """
        self.validate()
        for i, audio in enumerate(self.ground_truth):
            mono = audio.to_mono()
            self.ground_truth[i] = AudioSignal(audio_data_array=mono, sample_rate=self.sample_rate)

        for i, audio in enumerate(self.estimated_sources):
            mono = audio.to_mono()
            self.estimated_sources[i] = AudioSignal(audio_data_array=mono, sample_rate=self.sample_rate)

    def transform_sources_to_array(self):
        """

        Returns:

        """
        estimated_source_array = np.stack([x.audio_data for x in self.estimated_sources], axis=-1)
        reference_source_array = np.stack([x.audio_data for x in self.ground_truth], axis=-1)
        return reference_source_array, estimated_source_array

    def bss_eval_sources(self):
        """

        Returns:

        """
        self.validate()

    def bss_eval_images(self):
        """

        Returns:

        """
        self.validate()

    def bss_eval_sources_framewise(self):
        """

        Returns:

        """
        self.validate()

    def bss_eval_images_framewise(self):
        """

        Returns:

        """
        self.validate()
