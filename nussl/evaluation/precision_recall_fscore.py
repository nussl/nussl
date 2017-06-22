#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

F-score
"""

import sklearn

import evaluation_base
from nussl.separation.masks import binary_mask


class PrecisionRecallFScore(evaluation_base.EvaluationBase):
    """
    Precision, Recall, and F-Score
    """

    def __init__(self, true_sources_mask_list, estimated_sources_mask_list, source_labels=None):
        super(PrecisionRecallFScore, self).__init__(true_sources_list=true_sources_mask_list,
                                                    estimated_sources_list=estimated_sources_mask_list,
                                                    source_labels=source_labels)

    @staticmethod
    def _verify_input_list(mask_list):
        if isinstance(mask_list, binary_mask.BinaryMask):
            mask_list = [mask_list]
        elif isinstance(mask_list, list):
            if not all(isinstance(m, binary_mask.BinaryMask) for m in mask_list):
                raise ValueError('All items in mask_list must be of type nussl.BinaryMask!')

        if not all(mask_list[0].shape == m.shape for m in mask_list):
            raise ValueError('All masks must be the same shape!')

        return mask_list

    @staticmethod
    def _preprocess(mask1, mask2):
        """
        
        Args:
            mask1: 
            mask2: 

        Returns:

        """
        assert isinstance(mask1, binary_mask.BinaryMask)
        assert isinstance(mask2, binary_mask.BinaryMask)
        return mask1.mask.ravel(), mask2.mask.ravel()

    def _precision(self, true_mask, estimated_mask):
        """
        
        Args:
            true_mask: 
            estimated_mask: 

        Returns:

        """

        return sklearn.metrics.precision_score(*self._preprocess(true_mask, estimated_mask))

    def _recall(self, true_mask, estimated_mask):
        """
        
        Args:
            true_mask: 
            estimated_mask: 

        Returns:

        """
        return sklearn.metrics.recall_score(*self._preprocess(true_mask, estimated_mask))

    def _f_score(self, true_mask, estimated_mask):
        """
        
        Args:
            true_mask: 
            estimated_mask: 

        Returns:

        """
        return sklearn.metrics.f1_score(*self._preprocess(true_mask, estimated_mask))

    def _accuracy(self, true_mask, estimated_mask):
        """
        
        Args:
            true_mask: 
            estimated_mask: 

        Returns:

        """
        return sklearn.metrics.accuracy_score(*self._preprocess(true_mask, estimated_mask))

    def evaluate(self):
        """
        
        Returns:

        """
        for i, true_mask in enumerate(self.true_sources_list):
            est_mask = self.estimated_sources_list[i]

            label = self.source_labels[i]
            results = {'Accuracy': self._accuracy(true_mask, est_mask),
                       'Precision': self._precision(true_mask, est_mask),
                       'Recall': self._recall(true_mask, est_mask),
                       'F1-Score': self._f_score(true_mask, est_mask)}

            self.scores[label] = results

        return self.scores
