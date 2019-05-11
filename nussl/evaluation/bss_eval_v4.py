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
import numpy as np

from . import bss_eval_base
from ..core import constants, utils


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wrapper class for ``museval`` implementation of the BSS-Eval metrics (SDR, SIR, SAR).
Contains logic for loading ground truth AudioSignals and estimated
AudioSignals to compute BSS-Eval metrics. The ``mir_eval`` module contains an
implementation of BSS-Eval version 4.
"""


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

    def __init__(self, true_sources, estimated_sources,
                 mode='v4', output_dir=None, win=2.0, hop=1.5,
                 source_labels = None,
                 compute_permutation=False):

        if type(true_sources) is not type(estimated_sources):
            raise bss_eval_base.BssEvalException('true_sources and estimated_sources must both be '
                                                 'lists or both be dicts!')

        have_list = type(true_sources) is list

        self._scores = {}
        self.mode = mode
        self.output_dir = output_dir
        self.win = win
        self.hop = hop
        self.compute_permutation = compute_permutation

        # Set up the dictionaries for museval
        # self.true_sources is filled with AudioSignals (b/c nussl converts it to a Track)
        # & self.estimates is raw numpy arrays
        # Assume they know what they're doing...
        self.true_sources_list = true_sources
        self.estimated_sources_list = estimated_sources
        self.sample_rate = self.true_sources_list[0].sample_rate
        self.num_channels = self.true_sources_list[0].num_channels
        self.source_labels = source_labels
        if self.source_labels is None:
            self.source_labels = [x.path_to_input_file.split('/')[-1] for x in self.true_sources_list]


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

    def _preprocess_sources(self):
        """
        Prepare the :ref:`audio_data` in the sources for ``mir_eval``.
        Returns:
            (:obj:`np.ndarray`, :obj:`np.ndarray`) reference_source_array, estimated_source_array

        """
        reference_source_array = np.stack([np.copy(x.audio_data.T)
                                            for x in self.true_sources_list], axis=0)
        estimated_source_array = np.stack([np.copy(x.audio_data.T)
                                            for x in self.estimated_sources_list], axis=0)

        return reference_source_array, estimated_source_array

    def evaluate(self):
        self.validate()
        reference, estimated = self._preprocess_sources()
        if len(reference.shape) < 3:
            reference = np.expand_dims(reference, axis=-1)
            estimated = np.expand_dims(estimated, axis=-1)
            
        bss_output = museval.metrics.bss_eval(
            reference, estimated, window=int(self.sample_rate * self.win), 
            hop=int(self.sample_rate * self.hop), 
            compute_permutation = self.compute_permutation
        )

        self._populate_scores_dict(bss_output)

        return self.scores

    def _populate_scores_dict(self, bss_output):
        
        sdr_list, isr_list, sir_list, sar_list, perm = bss_output  # Unpack
        self.scores[self.RAW_VALUES] = {self.SDR: sdr_list, self.ISR: isr_list, self.SIR: sir_list,
                                        self.SAR: sar_list, self.PERMUTATION: perm}

        idx = 0
        for i, label in enumerate(self.source_labels):
            self.scores[label] = {}

            self.scores[label][self.SDR] = sdr_list[perm[idx]]
            self.scores[label][self.ISR] = isr_list[perm[idx]]
            self.scores[label][self.SIR] = sir_list[perm[idx]]
            self.scores[label][self.SAR] = sar_list[perm[idx]]
            idx += 1

        self.scores[self.PERMUTATION] = perm