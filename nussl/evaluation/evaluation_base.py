#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EvaluationBase is the base class for all evaluation methods in nussl.
"""
import nussl.utils
from nussl.audio_signal import AudioSignal


class EvaluationBase(object):
    """
    Base class for all Evaluation classes for source separation algorithm in nussl. 
    Contains common functions for all evaluation techniques. This class should not be instantiated directly.
    
    Both `true_sources_list` and `estimated_sources_list` get validated using the private method
    `EvaluationBase._verify_input_list()`. If your evaluation needs to verify that input is set correctly (recommended)
    overwrite that method to add checking.
    
    Args:
        true_sources_list (list): List of objects that contain one ground truth source per object. In some instances
            (such as the BSSEval-objects) this list is filled with :ref:`AudioSignals` but in other cases it is 
            populated with :ref:`MaskBase` -derived objects (i.e., either a :ref:`BinaryMask` 
            or :ref:`SoftMask` object). 
        estimated_sources_list (list): List of objects that contain source estimations from a source separation 
            algorithm. List should be populated with the same type of objects and in the same order as 
            `true_sources_list`.
        source_labels (list, Optional): List of strings that are labels for each source to be used as keys for 
            the scores. Default value is `None` and in that case labels are `Source 0`, `Source 1`, etc.
        do_mono (bool, Optional): Whether to make the objects in `true_sources_list` and `estimated_sources_list` mono
            prior to doing the evaluation. This may not work for all `EvaluationBase`-derived objects.
        **kwargs: Additional arguments for subclasses.
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
        """
        Base method for verifying a list of input objects for an `EvaluationBased`-derived object. Override this
        method when creating new `EvaluationBased`-derived class.
        
        By default calls :ref:`nussl.utils._verify_audio_signal_list_strict`, which verifies that all objects in the
        input list are :ref:`AudioSignal` objects with the same length, sample rate and have identical 
        number of channels.
        
        Args:
            audio_signal_list (list): List of objects that contain one signal per object. In some instances
            (such as the BSSEval-objects) this list is filled with :ref:`AudioSignals` but in other cases it is 
            populated with :ref:`MaskBase` -derived objects (i.e., either a :ref:`BinaryMask` 
            or :ref:`SoftMask` object). In the latter case, this method is overridden with a specific function 
            in :ref:`PrecisionRecallFScore`.

        Returns:
            A verified list of objects that are ready for running the evaluation method.

        """
        return nussl._verify_audio_signal_list_strict(audio_signal_list)

    def evaluate(self):
        """
        This function runs the evaluation method. Do not call this directly from `EvaluationBase`
        
        Raises:
            NotImplementedError

        """
        raise NotImplementedError('Cannot call base class!')

    @property
    def scores(self):
        """
        A dictionary that stores all scores from the evaluation method. Gets populated when :ref:`evaluate` gets run.

        """
        return self._scores
