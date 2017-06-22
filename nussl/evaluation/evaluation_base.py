#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EvaluationBase is the base class for all evaluation methods in nussl.
"""
import nussl.utils
from nussl.audio_signal import AudioSignal


class EvaluationBase(object):
    """
    Base class for all Evaluation classes. Contains common functions for all evaluation techniques.
    """

    def __init__(self, true_sources_list, estimated_sources_list, source_labels=None, **kwargs):
        self.true_sources_list = self._verify_input_list(true_sources_list)
        self.estimated_sources_list = self._verify_input_list(estimated_sources_list)

        if len(self.true_sources_list) != len(self.estimated_sources_list):
            raise ValueError('Must have the same number of objects in true_sources_list and estimated_sources_list!')

        # set the labels up correctly
        if source_labels is None:
            self.source_labels = ['Source {}'.format(i) for i in range(len(true_sources_list))]
        else:
            assert isinstance(source_labels, list), 'Expected source_labels to be a list of strings!'
            if not all([isinstance(l, str) for l in source_labels]):
                raise ValueError('All labels must be strings!')

            # Can't use a source_labels list longer than the sources
            if len(source_labels) > len(self.true_sources_list):
                raise ValueError('Labels list is longer than sources list!')

            # If not all sources have labels, we'll just give them generic ones...
            if len(source_labels) < len(self.true_sources_list):
                start, stop = len(source_labels), len(self.true_sources_list)
                for i in range(start, stop):
                    source_labels.append('Source {}'.format(i))

            self.source_labels = source_labels

        self.evaluation_object_type = type(self.true_sources_list[0])
        self.do_mono = None
        self._scores = {}

        if 'do_mono' in kwargs:
            self.do_mono = kwargs['do_mono']

            if self.evaluation_object_type is AudioSignal:
                if self.do_mono:
                    self.num_channels = 1
                    [g.to_mono(overwrite=True) for g in true_sources_list]
                    [e.to_mono(overwrite=True) for e in estimated_sources_list]
                else:
                    self.num_channels = true_sources_list[0].num_channels

    @staticmethod
    def _verify_input_list(audio_signal_list):
        return nussl.utils._verify_audio_signal_list_strict(audio_signal_list)

    def evaluate(self):
        """
        This function runs the evaluation method
        Returns:

        """
        raise NotImplementedError('Cannot call base class!')

    @property
    def scores(self):
        """
        
        Returns:

        """
        return self._scores
