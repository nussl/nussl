#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

import museval

import bss_eval_base


class BSSEvalImages(bss_eval_base.BSSEvalBase):
    """
    Wrapper class for ``mir_eval`` implementation of the BSS-Eval Iamges metrics (SDR, SIR, SAR).
    Contains logic for loading ground truth AudioSignals and estimated
    AudioSignals to compute BSS-Eval Images metrics. The ``mir_eval`` module contains
    an implementation of BSS-Eval version 3.

    The BSS-Eval metrics attempt to measure perceptual quality by comparing sources
    estimated from a source separation algorithm to the ground truth, known sources.
    These metrics evaluate the distortion (SDR) and artifacts (SAR) present in the
    estimated signals as well as the interference (SIR) from other sources in a given
    estimated source. Results are returned in units of dB, with higher values indicating
    better quality.

    See Also:
        * For more information on ``mir_eval`` (python implementation of BSS-Eval v3) see
        `its Github page <https://github.com/craffel/mir_eval>`_.

        * For more information on the BSS-Eval metrics, see the webpage for
        `the original MATLAB implementation <http://bass-db.gforge.inria.fr/bss_eval/>`_.

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

    def __init__(self, true_sources_list, estimated_sources_list, source_labels=None,
                 algorithm_name=None, do_mono=False, compute_permutation=True):
        super(BSSEvalImages, self).__init__(true_sources_list=true_sources_list,
                                            estimated_sources_list=estimated_sources_list,
                                            source_labels=source_labels, do_mono=do_mono,
                                            compute_permutation=compute_permutation)

        self._mir_eval_func = museval.metrics.bss_eval_images

    def _preprocess_sources(self):
        reference, estimated = super(BSSEvalImages, self)._preprocess_sources()

        if self.num_channels == 1:
            raise Exception("Can't run bss_eval_images on mono audio signals!")

        museval.metrics.validate(reference, estimated)

        return reference, estimated

    def _populate_scores_dict(self, bss_output):
        sdr_list, isr_list, sir_list, sar_list, perm = bss_output  # Unpack
        assert len(sdr_list) == len(sir_list) \
               == len(sar_list) == len(isr_list) == len(self.true_sources_list) * self.num_channels

        self.scores[self.RAW_VALUES] = {self.SDR: sdr_list, self.ISR: isr_list, self.SIR: sir_list,
                                        self.SAR: sar_list, self.PERMUTATION: perm}

        idx = 0
        for i, label in enumerate(self.source_labels):
            self.scores[label] = {}
            for ch in range(self.num_channels):
                chan = 'Ch {}'.format(ch)
                self.scores[label][chan] = {}

                self.scores[label][chan][self.SDR] = sdr_list[perm[idx]]
                self.scores[label][chan][self.ISR] = isr_list[perm[idx]]
                self.scores[label][chan][self.SIR] = sir_list[perm[idx]]
                self.scores[label][chan][self.SAR] = sar_list[perm[idx]]
                idx += 1

        self.scores[self.PERMUTATION] = perm
