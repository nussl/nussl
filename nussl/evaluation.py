#!/usr/bin/env python
# -*- coding: utf-8 -*-

from audio_signal import AudioSignal
from mir_eval import separation
import numpy as np
import json

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
    def __init__(self, ground_truth, estimated_sources=None, ground_truth_labels=None, algorithm_name=None,
                 do_mono=False, compute_permutation=True, hop_size=15, segment_size=30):
        # Do input checking
        ground_truth = self._verify_input_list(ground_truth, do_mono)
        if estimated_sources is not None:
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

        if ground_truth_labels is None:
            self.ground_truth_labels = ['Source %d' % i for i in range(len(ground_truth))]
        else:
            self.ground_truth_labels = ground_truth_labels
        if algorithm_name is None:
            self._algorithm_name = 'Approach'
        else:
            assert (type(algorithm_name) == str)
            self._algorithm_name = algorithm_name
        self.estimated_sources = estimated_sources
        self.sample_rate = ground_truth[0].sample_rate
        
        self.compute_permutation = compute_permutation

        self.segment_size = segment_size
        self.hop_size = hop_size
        self.scores = {}

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
            # I think we need it. If something about the signals match, it'll be pointless to get something like the
            # SDR.
            raise ValueError('Not all AudioSignal objects have the same sample rate.')

        if not do_mono:
            if not all(audio_signal_list[0].num_channels == g.num_channels for g in audio_signal_list):
                raise ValueError('Not all AudioSignal objects have the number of channels.')

        return audio_signal_list

    @property
    def algorithm_name(self):
        return self._algorithm_name

    @algorithm_name.setter
    def algorithm_name(self, value):
        assert (type(value) == str)
        self._algorithm_name = value
        self.scores[self._algorithm_name] = {}

    def validate(self):
        """

        Returns:

        """
        if self.estimated_sources is None:
            raise ValueError('Must set estimated_sources first!')
        estimated_lengths = [x.signal_length for x in self.estimated_sources]
        reference_lengths = [x.signal_length for x in self.ground_truth]

        if len(set(estimated_lengths)) > 1:
            raise Exception('All AudioSignals in estimated_sources must be the same length!')
        if len(set(reference_lengths)) > 1:
            raise Exception('All AudioSignals in ground_truth must be the same length!')
    

    def transform_sources_to_array(self):
        """

        Returns:

        """
        estimated_source_array = np.swapaxes(np.stack([np.copy(x.audio_data) for x in self.ground_truth], axis=-1),
                                             0, -1)
        reference_source_array = np.swapaxes(np.stack([np.copy(x.audio_data) for x in self.estimated_sources], axis=-1),
                                             0, -1)

        return reference_source_array, estimated_source_array

    def bss_eval_sources(self):
        """

        Returns:

        """
        self.validate()
        reference, estimated = self.transform_sources_to_array()

        if self.num_channels != 1:
            reference = np.sum(reference, axis=-1)
            estimated = np.sum(estimated, axis=-1)
        separation.validate(reference, estimated)
        sdr, sir, sar, perm = separation.bss_eval_sources(reference, estimated,
                                                          compute_permutation = self.compute_permutation)
        self.scores[self.algorithm_name]['Sources'] = {}
        self.scores[self.algorithm_name]['Sources']['Source to Distortion'] = sdr.tolist()
        self.scores[self.algorithm_name]['Sources']['Source to Interference'] = sir.tolist()
        self.scores[self.algorithm_name]['Sources']['Source to Artifact'] = sar.tolist()
        self.scores[self.algorithm_name]['Sources']['Permutation'] = perm.tolist()
        self.scores[self.algorithm_name]['Sources']['Labels'] = [self.ground_truth_labels[i] for i in perm.tolist()]

    def bss_eval_images(self):
        """

        Returns:

        """
        self.validate()
        if self.num_channels == 1:
            raise Exception("Can't run bss_eval_images on mono audio signals!")
        reference, estimated = self.transform_sources_to_array()
        separation.validate(reference, estimated)
        sdr, isr, sir, sar, perm = separation.bss_eval_images(reference, estimated,
                                                          compute_permutation=self.compute_permutation)
        self.scores[self.algorithm_name]['Images'] = {}
        self.scores[self.algorithm_name]['Images']['Source to Distortion'] = sdr.tolist()
        self.scores[self.algorithm_name]['Images']['Image to Spatial'] = isr.tolist()
        self.scores[self.algorithm_name]['Images']['Source to Interference'] = sir.tolist()
        self.scores[self.algorithm_name]['Images']['Source to Artifact'] = sar.tolist()
        self.scores[self.algorithm_name]['Images']['Permutation'] = perm.tolist()
        self.scores[self.algorithm_name]['Images']['Labels'] = [self.ground_truth_labels[i] for i in perm.tolist()]

    def bss_eval_sources_framewise(self):
        """

        Returns:

        """
        self.validate()
        reference, estimated = self.transform_sources_to_array()
        if self.num_channels != 1:
            reference = np.sum(reference, axis=-1)
            estimated = np.sum(estimated, axis=-1)
        separation.validate(reference, estimated)
        sdr, sir, sar, perm = separation.bss_eval_sources_framewise(reference, estimated,
                                                        window = self.segment_size, hop = self.hop_size,
                                                        compute_permutation=self.compute_permutation)
        self.scores[self.algorithm_name]['Source frames'] = {}
        self.scores[self.algorithm_name]['Source frames']['Source to Distortion'] = sdr.tolist()
        self.scores[self.algorithm_name]['Source frames']['Source to Interference'] = sir.tolist()
        self.scores[self.algorithm_name]['Source frames']['Source to Artifact'] = sar.tolist()
        self.scores[self.algorithm_name]['Source frames']['Permutation'] = perm.tolist()
        self.scores[self.algorithm_name]['Source frames']['Labels'] = [self.ground_truth_labels[i] for i in perm.tolist()]

    def bss_eval_images_framewise(self):
        """

        Returns:

        """
        self.validate()
        if self.num_channels == 1:
            raise Exception("Can't run bss_eval_Image frames_framewise on mono audio signals!")
        reference, estimated = self.transform_sources_to_array()
        separation.validate(reference, estimated)
        sdr, isr, sir, sar, perm = separation.bss_eval_images_framewise(reference, estimated,
                                                            window=self.segment_size, hop=self.hop_size,
                                                            compute_permutation=self.compute_permutation)
        print perm
        self.scores[self.algorithm_name]['Image frames'] = {}
        self.scores[self.algorithm_name]['Image frames']['Source to Distortion'] = sdr.tolist()
        self.scores[self.algorithm_name]['Image frames']['Image to Spatial'] = isr.tolist()
        self.scores[self.algorithm_name]['Image frames']['Source to Interference'] = sir.tolist()
        self.scores[self.algorithm_name]['Image frames']['Source to Artifact'] = sar.tolist()
        self.scores[self.algorithm_name]['Image frames']['Permutation'] = perm.tolist()
        self.scores[self.algorithm_name]['Image frames']['Labels'] = [self.ground_truth_labels[i] for i in perm.tolist()]

    def write_scores_to_file(self, filename):
        f = open(filename, 'w')
        f.write(self.to_json())
        f.close()

    def to_json(self):
        return json.dumps(self.scores, sort_keys = True,
                          indent=4, separators=(',', ': '))
