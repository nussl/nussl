#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base class for both BSS Eval algorithms (:ref:`BSSEvalSources` and :ref:`BSSEvalImages`). Contains
most of the logic for these base classes.
"""

import numpy as np

import evaluation_base

# TODO: json
# TODO: framewise
# TODO: Docs


class BSSEvalBase(evaluation_base.EvaluationBase):
    """Lets you load ground truth AudioSignals and estimated AudioSignals and compute separation
    evaluation criteria (SDR, SIR, SAR and delta SDR, delta SIR, delta SAR).

    Parameters:

    Examples:
  
    """
    SDR = 'SDR'
    SIR = 'SIR'
    SAR = 'SAR'
    ISR = 'ISR'
    PERMUTATION = 'permutation'
    RAW_VALUES = 'raw_values'

    def __init__(self, true_sources_list, estimated_sources_list, source_labels=None, algorithm_name=None,
                 do_mono=False, compute_permutation=True):
        super(BSSEvalBase, self).__init__(true_sources_list=true_sources_list,
                                          estimated_sources_list=estimated_sources_list,
                                          source_labels=source_labels, do_mono=do_mono)

        if algorithm_name is None:
            self._algorithm_name = 'Approach'
        else:
            assert type(algorithm_name) == str
            self._algorithm_name = algorithm_name

        self.compute_permutation = compute_permutation
        self._mir_eval_func = None

    @property
    def algorithm_name(self):
        """
        Name of the algorithm that is being evaluated
        Returns:

        """
        return self._algorithm_name

    @algorithm_name.setter
    def algorithm_name(self, value):
        assert type(value) == str
        self._algorithm_name = value
        self.scores[self._algorithm_name] = {label: {} for label in self.source_labels}

    def validate(self):
        """

        Returns:

        """
        # TODO: This might be obsolete
        if self.estimated_sources_list is None:
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
        estimated_source_array = np.vstack([np.copy(x.audio_data) for x in self.true_sources_list])
        reference_source_array = np.vstack([np.copy(x.audio_data) for x in self.estimated_sources_list])

        return reference_source_array, estimated_source_array

    def evaluate(self):
        """

        Returns:

        """
        self.validate()
        reference, estimated = self._preprocess_sources()

        if self._mir_eval_func is None:
            raise NotImplementedError('Cannot call base class! Try calling BSSEvalSources or BSSEvalImages')

        bss_output = self._mir_eval_func(reference, estimated, compute_permutation=self.compute_permutation)

        self._populate_scores_dict(bss_output)

        return self.scores

    def _populate_scores_dict(self, bss_output):
        """
        Populates the scores dict from the
        Args:
            bss_output:

        Returns:

        """


    # def bss_eval_sources_framewise(self):
    #     """
    #     TODO - figure out compute_permutation=True branch will work here
    #     Returns:
    #
    #     """
    #     raise NotImplementedError("Still working on this!")
    #     self.validate()
    #     reference, estimated = self._preprocess_sources()
    #     if self.num_channels != 1:
    #         reference = np.sum(reference, axis=-1)
    #         estimated = np.sum(estimated, axis=-1)
    #     separation.validate(reference, estimated)
    #     sdr, sir, sar, perm = separation.bss_eval_sources_framewise(reference, estimated,
    #                                                     window = self.segment_size, hop = self.hop_size,
    #                                                     compute_permutation=self.compute_permutation)
    #
    # def bss_eval_images_framewise(self):
    #     """
    #     TODO - figure out compute_permutation=True branch will work here
    #     Returns:
    #
    #     """
    #     raise NotImplementedError("Still working on this!")
    #     self.validate()
    #     if self.num_channels == 1:
    #         raise Exception("Can't run bss_eval_Image frames_framewise on mono audio signals!")
    #     reference, estimated = self._preprocess_sources()
    #     separation.validate(reference, estimated)
    #     sdr, isr, sir, sar, perm = separation.bss_eval_images_framewise(reference, estimated,
    #                                                         window=self.segment_size, hop=self.hop_size,
    #                                                         compute_permutation=self.compute_permutation)
