#!/usr/bin/env python
# -*- coding: utf-8 -*-
import mir_eval
import numpy as np
import json

import evaluation_base


class BSSEvalBase(evaluation_base.EvaluationBase):
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
    def __init__(self, true_sources_list, estimated_sources_list, source_labels=None, algorithm_name=None,
                 do_mono=False, compute_permutation=True):
        super(BSSEvalBase, self).__init__(true_sources_list=true_sources_list,
                                          estimated_sources_list=estimated_sources_list,
                                          source_labels=source_labels, do_mono=do_mono)

        if algorithm_name is None:
            self._algorithm_name = 'Approach'
        else:
            assert (type(algorithm_name) == str)
            self._algorithm_name = algorithm_name

        
        self.compute_permutation = compute_permutation


    @property
    def algorithm_name(self):
        """
        Name of the algorithm that is being evaluated
        Returns:

        """
        return self._algorithm_name

    @algorithm_name.setter
    def algorithm_name(self, value):
        assert (type(value) == str)
        self._algorithm_name = value
        self.scores[self._algorithm_name] = {label: {} for label in self.source_labels}

    def validate(self):
        """

        Returns:

        """
        if self.estimated_sources is None:
            raise ValueError('Must set estimated_sources first!')
        estimated_lengths = [x.signal_length for x in self.estimated_sources_list]
        reference_lengths = [x.signal_length for x in self.true_sources_list]

        if len(set(estimated_lengths)) > 1:
            raise Exception('All AudioSignals in estimated_sources must be the same length!')
        if len(set(reference_lengths)) > 1:
            raise Exception('All AudioSignals in ground_truth must be the same length!')
    
    def _preprocess_sources(self):
        """

        Returns:

        """
        estimated_source_array = np.swapaxes(np.stack([np.copy(x.audio_data) for x in self.true_sources_list], axis=-1),
                                             0, -1)
        reference_source_array = np.swapaxes(np.stack([np.copy(x.audio_data) for x in self.estimated_sources_list], axis=-1),
                                             0, -1)

        return reference_source_array, estimated_source_array

    def bss_eval_sources(self):
        """

        Returns:

        """
        self.validate()
        reference, estimated = self._preprocess_sources()

        if self.num_channels != 1:
            reference = np.sum(reference, axis=-1)
            estimated = np.sum(estimated, axis=-1)
        mir_eval.separation.validate(reference, estimated)
        sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(reference, estimated,
                                                                   compute_permutation=self.compute_permutation)

        for i, label in enumerate(self.source_labels):
            self.scores[self.algorithm_name][label]['Sources'] = {}

            D = self.scores[self.algorithm_name][label]['Sources']

            D['Source to Distortion'] = sdr.tolist()[i]
            D['Source to Interference'] = sir.tolist()[i]
            D['Source to Artifact'] = sar.tolist()[i]

        self.scores[self.algorithm_name]['Permutation'] = perm.tolist()

    def bss_eval_images(self):
        """

        Returns:

        """
        self.validate()
        if self.num_channels == 1:
            raise Exception("Can't run bss_eval_images on mono audio signals!")
        reference, estimated = self._preprocess_sources()
        mir_eval.separation.validate(reference, estimated)
        sdr, isr, sir, sar, perm = mir_eval.separation.bss_eval_images(reference, estimated,
                                                                       compute_permutation=self.compute_permutation)

        for i, label in enumerate(self.source_labels):
            self.scores[self.algorithm_name][label]['Images'] = {}

            D = self.scores[self.algorithm_name][label]['Images']

            D['Source to Distortion'] = sdr.tolist()[i]
            D['Image to Spatial'] = isr.tolist()[i]
            D['Source to Interference'] = sir.tolist()[i]
            D['Source to Artifact'] = sar.tolist()[i]

        self.scores[self.algorithm_name]['Permutation'] = perm.tolist()

    def bss_eval_sources_framewise(self):
        """
        TODO - figure out compute_permutation=True branch will work here
        Returns:

        """
        raise NotImplementedError("Still working on this!")
        self.validate()
        reference, estimated = self._preprocess_sources()
        if self.num_channels != 1:
            reference = np.sum(reference, axis=-1)
            estimated = np.sum(estimated, axis=-1)
        separation.validate(reference, estimated)
        sdr, sir, sar, perm = separation.bss_eval_sources_framewise(reference, estimated,
                                                        window = self.segment_size, hop = self.hop_size,
                                                        compute_permutation=self.compute_permutation)

    def bss_eval_images_framewise(self):
        """
        TODO - figure out compute_permutation=True branch will work here
        Returns:

        """
        raise NotImplementedError("Still working on this!")
        self.validate()
        if self.num_channels == 1:
            raise Exception("Can't run bss_eval_Image frames_framewise on mono audio signals!")
        reference, estimated = self._preprocess_sources()
        separation.validate(reference, estimated)
        sdr, isr, sir, sar, perm = separation.bss_eval_images_framewise(reference, estimated,
                                                            window=self.segment_size, hop=self.hop_size,
                                                            compute_permutation=self.compute_permutation)

    def load_scores_from_file(self, filename):
        f = open(filename, 'r')
        self.scores = json.load(f)
        f.close()

    def write_scores_to_file(self, filename):
        f = open(filename, 'w')
        f.write(self.to_json())
        f.close()

    def to_json(self):
        return json.dumps(self.scores, sort_keys=True,
                          indent=4, separators=(',', ': '))
