#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO
"""
import numpy as np
import museval

import bss_eval_base


class BSSEvalV4(bss_eval_base.BSSEvalBase):
    """

    """
    SDR_FRAMES = 'SDR_Frames'
    SAR_FRAMES = 'SAR_Frames'
    SIR_FRAMES = 'SIR_Frames'
    ISR_FRAMES = 'ISR_Frames'

    def __init__(self, true_sources_list, estimated_sources_list, source_labels=None,
                 algorithm_name=None, do_mono=False, compute_permutation=True):
        super(BSSEvalV4, self).__init__(true_sources_list=true_sources_list,
                                        estimated_sources_list=estimated_sources_list,
                                        source_labels=source_labels, do_mono=do_mono,
                                        compute_permutation=compute_permutation)

        self._mir_eval_func = museval.metrics.bss_eval_images_framewise

    def _preprocess_sources(self):
        # reference, estimated = super(BSSEvalV4, self)._preprocess_sources()
        reference = np.dstack((self.true_sources_list[i].audio_data.T
                               for i in range(len(self.true_sources_list)))).transpose((2, 0, 1))
        estimated = np.dstack((self.estimated_sources_list[i].audio_data.T
                               for i in range(len(self.estimated_sources_list))))
        estimated = estimated.transpose((2, 0, 1))

        # if self.num_channels == 1:
        #     raise Exception("Can't run bss_eval_images on mono audio signals!")

        museval.metrics.validate(reference, estimated)

        return reference, estimated

    def _populate_scores_dict(self, bss_output):

        # Store them as list for ease of json
        sdr_list, isr_list, sir_list, sar_list, perm = map(lambda l: l.tolist(), bss_output)
        self.scores[self.RAW_VALUES] = {self.SDR: sdr_list, self.ISR: isr_list, self.SIR: sir_list,
                                        self.SAR: sar_list,
                                        self.PERMUTATION: perm}

        sdr_list, isr_list, sir_list, sar_list, perm = bss_output
        n_frames = sdr_list.shape[1]

        for i, label in enumerate(self.source_labels):
            self.scores[label] = {}
            self.scores[label][self.SDR_FRAMES] = []
            self.scores[label][self.ISR_FRAMES] = []
            self.scores[label][self.SIR_FRAMES] = []
            self.scores[label][self.SAR_FRAMES] = []
            for f in range(n_frames):

                self.scores[label][self.SDR_FRAMES].append(sdr_list[i, f])
                self.scores[label][self.ISR_FRAMES].append(isr_list[i, f])
                self.scores[label][self.SIR_FRAMES].append(sir_list[i, f])
                self.scores[label][self.SAR_FRAMES].append(sar_list[i, f])

        self.scores[self.PERMUTATION] = perm.tolist()
