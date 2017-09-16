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

        self.mir_eval_func = mir_eval.separation.bss_eval_images


    def _preprocess_sources(self):
        reference, estimated = super(BSSEvalImages, self)._preprocess_sources()

        if self.num_channels == 1:
            raise Exception("Can't run bss_eval_images on mono audio signals!")

        mir_eval.separation.validate(reference, estimated)

        return reference, estimated

