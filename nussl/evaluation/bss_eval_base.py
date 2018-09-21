#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base class for both BSS Eval algorithms (:ref:`BSSEvalSources` and :ref:`BSSEvalImages`). Contains
most of the logic for these base classes.
"""

import numpy as np

import evaluation_base


class BSSEvalBase(evaluation_base.EvaluationBase):
    """
    Base class for ``mir_eval`` implementation of the BSS-Eval metrics (SDR, SIR, SAR).
    Contains logic for loading ground truth :class:`AudioSignal`s and estimated
    :class:`AudioSignal`s to compute BSS-Eval metrics. The ``mir_eval`` module contains
    an implementation of BSS-Eval version 3.

    The BSS-Eval metrics attempt to measure perceptual quality by comparing sources
    estimated from a source separation algorithm to the ground truth, known sources.
    These metrics evaluate the distortion (SDR) and artifacts (SAR) present in the
    estimated signals as well as the interference (SIR) from other sources in a given
    estimated source. Results are returned in units of dB, with higher values indicating
    better quality.

    See Also:
        * For more information on ``mir_eval`` (python implementation of BSS-Eval v3) see
        `its Github page<https://github.com/craffel/mir_eval>`.
        * For more information on the BSS-Eval metrics, see the webpage for
        `the original MATLAB implementation<http://bass-db.gforge.inria.fr/bss_eval/>`.
        * Implementations of this base class: :class:`BSSEvalSources` and :class:`BSSEvalImages`.
        * :class:`BSSEvalV4` for the ``museval`` version 4 BSS-Eval implementation.

    References:
        * Emmanuel Vincent, Rémi Gribonval, Cédric Févotte. Performance measurement in blind
        audio source separation. IEEE Transactions on Audio, Speech and Language Processing,
        Institute of Electrical and Electronics Engineers, 2006, 14 (4), pp.1462–1469.
        <inria-00544230>
        * Colin Raffel, Brian McFee, Eric J. Humphrey, Justin Salamon, Oriol Nieto, Dawen Liang,
        and Daniel P. W. Ellis, "mir_eval: A Transparent Implementation of Common MIR Metrics",
        Proceedings of the 15th International Conference on Music Information Retrieval, 2014.

    Args:
        true_sources_list (list): List of :class:`AudioSignal` objects that contain the ground
            truth sources for the mixture.
        estimated_sources_list (list):  List of :class:`AudioSignal` objects that contain estimate
            sources, output from source separation algorithms.
        source_labels (list): List of strings that are labels for each source to be used as keys for
            the scores. Default value is ``None`` and in that case labels are ``Source 0``,
            ``Source 1``, etc.
        algorithm_name (str): Name of the algorithm if using this object to compute many
            BSS-Eval metrics. Can be changed later.
        do_mono (bool): Should flatten the audio to mono before calculating metrics.
        compute_permutation (bool): Should try to find the best permutation for the estimated
            sources.

    """
    SDR = 'SDR'
    SIR = 'SIR'
    SAR = 'SAR'
    ISR = 'ISR'
    PERMUTATION = 'permutation'
    RAW_VALUES = 'raw_values'

    def __init__(self, true_sources_list, estimated_sources_list, source_labels=None,
                 algorithm_name=None, do_mono=False, compute_permutation=True):
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
        Name of the algorithm that is being evaluated.
        Returns:
            (str) Name of the algorithm being evaluated.
        """
        return self._algorithm_name

    @algorithm_name.setter
    def algorithm_name(self, value):
        assert type(value) == str
        self._algorithm_name = value
        self.scores[self._algorithm_name] = {label: {} for label in self.source_labels}

    def validate(self):
        """
        Checks to make sure the all of the input :class:`AudioSignal` objects have the
        same length.
        """
        # TODO: This might be obsolete
        if self.estimated_sources_list is None:
            raise BssEvalException('Must set estimated_sources first!')
        estimated_lengths = [x.signal_length for x in self.estimated_sources_list]
        reference_lengths = [x.signal_length for x in self.true_sources_list]

        if len(set(estimated_lengths)) > 1:
            raise BssEvalException('All AudioSignals in estimated_sources must be the same length!')
        if len(set(reference_lengths)) > 1:
            raise BssEvalException('All AudioSignals in ground_truth must be the same length!')
    
    def _preprocess_sources(self):
        """
        Prepare the :ref:`audio_data` in the sources for ``mir_eval``.
        Returns:
            (:obj:`np.ndarray`, :obj:`np.ndarray`) reference_source_array, estimated_source_array

        """
        estimated_source_array = np.vstack([np.copy(x.audio_data)
                                            for x in self.true_sources_list])
        reference_source_array = np.vstack([np.copy(x.audio_data)
                                            for x in self.estimated_sources_list])

        return reference_source_array, estimated_source_array

    def evaluate(self):
        """
        Actually runs the evaluation algorithm. Will be ``museval.metrics.bss_eval_images`` or
        ``museval.metrics.bss_eval_sources`` depending on which subclass is instantiated.
        Returns:
            (dict): Dictionary containing the scores.

        """
        self.validate()
        reference, estimated = self._preprocess_sources()

        if self._mir_eval_func is None:
            raise NotImplementedError('Cannot call base class! Try calling '
                                      'BSSEvalSources or BSSEvalImages')

        bss_output = self._mir_eval_func(reference, estimated,
                                         compute_permutation=self.compute_permutation)

        self._populate_scores_dict(bss_output)

        return self.scores

    def _populate_scores_dict(self, bss_output):
        """
        Formats and populates the :attr:`scores` dict from :func:`evaluate`.
        Args:
            bss_output (tuple): Direct output from the ``museval`` function.

        Returns:
            (dict) Reformatted dictionary from ``museval`` output.
        """


class BssEvalException(Exception):
    """
    Exception class for BSS-Eval
    """
    pass