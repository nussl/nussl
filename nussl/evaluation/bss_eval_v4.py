#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wrapper class for ``museval`` implementation of the BSS-Eval metrics (SDR, SIR, SAR).
Contains logic for loading ground truth AudioSignals and estimated
AudioSignals to compute BSS-Eval metrics. The ``mir_eval`` module contains an
implementation of BSS-Eval version 4.
"""
import museval
import json

import bss_eval_base
from ..core import constants, utils


class BSSEvalV4(bss_eval_base.BSSEvalBase):
    """
    Wrapper class for ``museval`` implementation of the BSS-Eval metrics (SDR, SIR, SAR).
    Contains logic for loading ground truth AudioSignals and estimated
    AudioSignals to compute BSS-Eval metrics. The ``mir_eval`` module contains an
    implementation of BSS-Eval version 4.

    The BSS-Eval metrics attempt to measure perceptual quality by comparing sources
    estimated from a source separation algorithm to the ground truth, known sources.
    These metrics evaluate the distortion (SDR) and artifacts (SAR) present in the
    estimated signals as well as the interference (SIR) from other sources in a given
    estimated source. Results are returned in units of dB, with higher values indicating
    better quality.

    Examples:
        :class:`BSSEvalV4` can be initialized in two ways, either with a list or a dict.
        See the example below for a demonstration:

        .. code-block:: python
            :linenos:

            mir1k_dir = 'path/to/MIR-1K'
            mix, vox, acc = next(nussl.datasets.mir1k(mir1k_dir))
            mix.to_mono(overwrite=True)

            r = nussl.RepetSim(mix)
            r()
            bg, fg = r.make_audio_signals()

            # Method 1: Dictionary where sources are explicit
            # Note that this dictionary exactly matches nussl.constants.VOX_ACC_DICT
            est_dict = {'vocals': fg, 'accompaniment': bg}
            gt_dict = {'vocals': vox, 'accompaniment': acc}
            bss = nussl.evaluation.BSSEvalV4(mix, gt_dict, est_dict)
            scores1 = bss.evaluate()

            # Method 2: List
            # Note that vocals are always expected to be first, then accompaniment.
            bss = nussl.evaluation.BSSEvalV4(mix, [vox, acc], [fg, bg])
            scores2 = bss.evaluate()

    See Also:
        * For more information on ``museval`` (python implementation of BSS-Eval v4) see
        `its Github page <https://github.com/sigsep/sigsep-mus-eval>`_ or
        'its documentation <https://sigsep.github.io/sigsep-mus-eval/>`_.

        * For more information on the BSS-Eval metrics, see the webpage for
        `the original MATLAB implementation <http://bass-db.gforge.inria.fr/bss_eval/>`_.

        * :class:`BSSEvalSources` and :class:`BSSEvalImages` for the ``mir_eval`` version 3
        BSS-Eval implementations.

    References:
        * Emmanuel Vincent, Rémi Gribonval, Cédric Févotte. Performance measurement in blind
        audio source separation. IEEE Transactions on Audio, Speech and Language Processing,
        Institute of Electrical and Electronics Engineers, 2006, 14 (4), pp.1462–1469.
        <inria-00544230>

        * Fabian-Robert Stöter, Antoine Liutkus, and Nobutaka Ito. The 2018 Signal Separation
        Evaluation Campaign. In International Conference on Latent Variable Analysis and Signal
        Separation, pages 293–305. Springer, 2018.

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
    SDR_MEANS = 'sdr_means'

    def __init__(self, mixture, true_sources, estimated_sources,
                 target_dict=constants.VOX_ACC_DICT,
                 mode='v4', output_dir=None, win=1.0, hop=1.0):
        # try:
        #     super(BSSEvalV4, self).__init__(true_sources_list=true_sources_list,
        #                                     estimated_sources_list=estimated_sources_list,
        #                                     source_labels=source_labels, do_mono=do_mono)
        # except evaluation_base.AudioSignalListMismatchError:
        #     pass

        # if vox_acc and target_dict == constants.VOX_ACC_DICT:
        #     self.source_labels = ['accompaniment', 'vocals']
        #
        # if target_dict == constants.STEM_TARGET_DICT:
        #     self.source_labels = ['drums', 'bass', 'other', 'vocals']
        #
        # self.true_sources = {l: self.true_sources_list[i] for i, l in enumerate(self.source_labels)}
        # self.estimates = {l: self.estimated_sources_list[i].audio_data.T
        #                        for i, l in enumerate(self.source_labels)}
        if type(true_sources) is not type(estimated_sources):
            raise bss_eval_base.BssEvalException('true_sources and estimated_sources must both be '
                                                 'lists or both be dicts!')

        have_list = type(true_sources) is list

        self._scores = {}
        self.target_dict = target_dict
        self.is_vox_acc = target_dict == constants.VOX_ACC_DICT
        self.is_stem = target_dict == constants.STEM_TARGET_DICT
        self.mixture = mixture
        self.mode = mode
        self.output_dir = output_dir
        self.win = win
        self.hop = hop

        # Set up the dictionaries for museval
        # self.true_sources is filled with AudioSignals (b/c nussl converts it to a Track)
        # & self.estimates is raw numpy arrays
        if self.is_vox_acc:
            if have_list:
                self.estimates = {'vocals': estimated_sources[0].audio_data.T,
                                  'accompaniment': estimated_sources[1].audio_data.T}
                self.true_sources = {'vocals': true_sources[0],
                                     'accompaniment': true_sources[1]}
            else:
                self.estimates = {'vocals': estimated_sources['vocals'].audio_data.T,
                                  'accompaniment': estimated_sources['accompaniment'].audio_data.T}
                self.true_sources = {'vocals': true_sources['vocals'],
                                     'accompaniment': true_sources['accompaniment']}
        else:
            # Assume they know what they're doing...
            self.true_sources = true_sources
            self.estimates = estimated_sources

            # TODO: STEM_TARGET_DICT logic

    def _get_scores(self, scores):
        s = scores.split()
        v, a = s[0], s[6].replace('\\n', '')
        i1, i2 = [2, 3, 4, 5], [8, 9, 10, 11]
        return {v: {self._parse(s[i])[0]: self._parse(s[i])[1] for i in i1},
                a: {self._parse(s[i])[0]: self._parse(s[i])[1] for i in i2}}

    @staticmethod
    def _parse(str_):
        bss_type, val = str_.split(':')
        val = float(val[:-3])
        return bss_type, val

    def _get_mean_scores(self, scores):
        return self._get_scores(repr(scores))

    def evaluate(self):
        track = utils.audio_signals_to_musdb_track(self.mixture, self.true_sources,
                                                   self.target_dict)

        bss_output = museval.eval_mus_track(track, self.estimates,
                                            output_dir=self.output_dir, mode=self.mode,
                                            win=self.win, hop=self.hop)

        self._populate_scores_dict(bss_output)

        return self.scores

    def _populate_scores_dict(self, bss_output):

        self.scores[self.RAW_VALUES] = json.loads(bss_output.json)  # Hack to format dict correctly
        self.scores[self.SDR_MEANS] = self._get_mean_scores(repr(bss_output))
