#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This class provides common statistical metrics for determining how well a source separation algorithm in nussl was
able to create a binary mask compared to a known binary mask. The metrics used here are 
`Precision, Recall <https://en.wikipedia.org/wiki/Precision_and_recall>`_,
`F-Score <https://en.wikipedia.org/wiki/F1_score>`_ (sometimes called F-measure or F1-score), and Accuracy
(though this is not reflected in the name of the class, it is simply   ``# correct / total``).


Examples:
     ::
    # Run Repet
    repet = nussl.Repet(mixture, mask_type=nussl.BinaryMask)  # it's important to specify BinaryMask!
    repet_masks = repet()
    
    # Get Ideal Binary Masks
    ideal_mask = nussl.IdealMask(mixture, [drums, flute], mask_type=nussl.BinaryMask)  # BinaryMask here, too!
    ideal_masks = ideal_mask()
    
    # Compare Repet to Ideal Binary Mask
    prf_repet = nussl.PrecisionRecallFScore(ideal_masks, repet_masks)
    prf_repet_scores = prf_repet.evaluate()

Scores for each source are stored in a nested dictionary aptly named ``scores``. This is a dictionary of dictionaries
where the key is the source label, and the value is another dictionary with scores for each of the metrics for that
source. So, for instance, the format of the ``prf_repet_scores`` dictionary from above is shown below:

.. code-block:: python

    {'Source 0' : {'Accuracy': 0.83,
                   'Precision': 0.78,
                   'Recall': 0.81,
                   'F1-Score': 0.77 },
     'Source 1' : {'Accuracy': 0.22,
                   'Precision': 0.12,
                   'Recall': 0.15,
                   'F1-Score': 0.19 }
    }


Notes:
    ``PrecisionRecallFScore`` can only be run using :ref:`binary_mask` objects. The constructor expects a list of 
    :ref:`binary_mask` objects for both the ground truth sources and the estimated sources.

See Also:
    * :ref:`evaluation_base` for more information about derived properties that this class has.
    
    * :ref:`ideal_mask` for information about how to get an array of ground truth binary masks.
    
    
    

"""

import sklearn

import evaluation_base
from nussl.separation.masks import binary_mask


class PrecisionRecallFScore(evaluation_base.EvaluationBase):
    """
    Args:
        true_sources_mask_list (list): List of :ref:`binary_mask` objects representing the ground truth sources.
        estimated_sources_mask_list (list): List of :ref:`binary_mask` objects representing the estimates from a source
         separation object
        source_labels (list) (Optional): List of ``str`` with labels for each source. If no labels are provided, sources
         will be labeled ``Source 0, Source 1, ...`` etc.
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
        Determines the precision, recall, f-score, and accuracy of each :ref:`binary_mask` object in 
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
