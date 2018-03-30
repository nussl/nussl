#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO
"""
import museval
import json

import bss_eval_base
from ..core import constants, utils


class BSSEvalV4(bss_eval_base.BSSEvalBase):
    """

    """
    SDR_MEANS = 'sdr_means'

    def __init__(self, mixture, true_sources_list, estimated_sources_list, source_labels=None,
                 do_mono=False, target_dict=constants.VOX_ACC_DICT, vox_acc=True,
                 mode='v4', output_dir=None, win=1.0, hop=1.0):
        super(BSSEvalV4, self).__init__(true_sources_list=true_sources_list,
                                        estimated_sources_list=estimated_sources_list,
                                        source_labels=source_labels, do_mono=do_mono)
        if vox_acc and target_dict == constants.VOX_ACC_DICT:
            self.source_labels = ['accompaniment', 'vocals']

        self.source_dict = {l: self.true_sources_list[i] for i, l in enumerate(self.source_labels)}
        self.estimates_dict = {l: self.estimated_sources_list[i].audio_data.T
                               for i, l in enumerate(self.source_labels)}
        self.target_dict = target_dict
        self.mixture = mixture
        self.mode = mode
        self.output_dir = output_dir
        self.win = win
        self.hop = hop

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
        track = utils.audio_signals_to_mudb_track(self.mixture, self.source_dict, self.target_dict)

        bss_output = museval.eval_mus_track(track, self.estimates_dict,
                                            output_dir=self.output_dir, mode=self.mode,
                                            win=self.win, hop=self.hop)

        self._populate_scores_dict(bss_output)

        return self.scores

    def _populate_scores_dict(self, bss_output):

        self.scores[self.RAW_VALUES] = json.loads(bss_output.json)  # Hack to format dict correctly
        self.scores[self.SDR_MEANS] = self._get_mean_scores(repr(bss_output))
