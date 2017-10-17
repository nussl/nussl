#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO
"""

import mir_eval

import bss_eval_base


class BSSEvalImages(bss_eval_base.BSSEvalBase):
    """

    """

    def __init__(self, true_sources_list, estimated_sources_list, source_labels=None, algorithm_name=None,
                 do_mono=False, compute_permutation=True):
        super(BSSEvalImages, self).__init__(true_sources_list=true_sources_list,
                                            estimated_sources_list=estimated_sources_list,
                                            source_labels=source_labels, do_mono=do_mono,
                                            compute_permutation=compute_permutation)

        self._mir_eval_func = mir_eval.separation.bss_eval_images

    def _preprocess_sources(self):
        reference, estimated = super(BSSEvalImages, self)._preprocess_sources()

        if self.num_channels == 1:
            raise Exception("Can't run bss_eval_images on mono audio signals!")

        mir_eval.separation.validate(reference, estimated)

        return reference, estimated

    def _populate_scores_dict(self, bss_output):
        sdr_list, isr_list, sir_list, sar_list, perm = bss_output  # Unpack
        assert len(sdr_list) == len(sir_list) \
               == len(sar_list) == len(isr_list) == len(self.true_sources_list) * self.num_channels

        self.scores[self.RAW_VALUES] = {self.SDR: sdr_list, self.ISR: isr_list, self.SIR: sir_list, self.SAR: sar_list,
                                        self.PERMUTATION: perm}

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
