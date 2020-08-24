import sklearn
import numpy as np

from . import EvaluationBase
from ..core.masks import BinaryMask


class PrecisionRecallFScore(EvaluationBase):
    """
    This class provides common statistical metrics for determining how well a source separation algorithm in nussl was
    able to create a binary mask compared to a known binary mask. The metrics used here are 
    `Precision, Recall <https://en.wikipedia.org/wiki/Precision_and_recall>`_,
    `F-Score <https://en.wikipedia.org/wiki/F1_score>`_ (sometimes called F-measure or F1-score), and Accuracy
    (though this is not reflected in the name of the class, it is simply   ``# correct / total``).

    Notes:
        * ``PrecisionRecallFScore`` can only be run using :ref:`binary_mask` objects. The constructor expects a list of 
        :ref:`binary_mask` objects for both the ground truth sources and the estimated sources.
        * ``PrecisionRecallFScore`` does not calculate the correct permutation of the estimated and ground truth sources;
        they are expected to be in the correct order when they are passed into ``PrecisionRecallFScore``.

    Args:
        true_sources_mask_list (list): List of :ref:`binary_mask` objects representing the ground truth sources.
        estimated_sources_mask_list (list): List of :ref:`binary_mask` objects representing the estimates from a source
         separation object
        source_labels (list) (Optional): List of ``str`` with labels for each source. If no labels are provided, sources
         will be labeled ``Source 0, Source 1, ...`` etc.        
    """

    def __init__(self, true_sources_list, estimated_sources_list, source_labels=None,
                 compute_permutation=False, best_permutation_key="F1-Score", **kwargs):
        self.true_sources_list = self._verify_input_list(true_sources_list)
        self.estimated_sources_list = self._verify_input_list(estimated_sources_list)

        super().__init__(true_sources_list, estimated_sources_list,
                         source_labels=source_labels,
                         compute_permutation=compute_permutation,
                         best_permutation_key=best_permutation_key, **kwargs)
        self.keys = ['Precision', 'Recall', 'Accuracy', 'F1-Score']

    @staticmethod
    def _verify_input_list(mask_list):
        if isinstance(mask_list, BinaryMask):
            mask_list = [mask_list]
        elif isinstance(mask_list, list):
            if not all(isinstance(m, BinaryMask) for m in mask_list):
                raise ValueError('All items in mask_list must be of type nussl.BinaryMask!')

        if not all(mask_list[0].shape == m.shape for m in mask_list):
            raise ValueError('All masks must be the same shape!')

        return mask_list

    def preprocess(self):
        n_channels = self.true_sources_list[0].num_channels
        references = np.stack(
            [np.copy(x.mask.reshape(-1, n_channels))
             for x in self.true_sources_list],
            axis=-1
        )
        estimates = np.stack(
            [np.copy(x.mask.reshape(-1, n_channels))
             for x in self.estimated_sources_list],
            axis=-1
        )
        return references, estimates

    def evaluate_helper(self, references, estimates, **kwargs):
        """
        Determines the precision, recall, f-score, and accuracy of each :ref:`binary_mask` object in 
        ``true_sources_mask_list`` and ``estimated_sources_mask_list``. Returns a list of results that is
        formatted like so:
        
        .. code-block:: python

            [              
                {'Accuracy': 0.83,
                'Precision': 0.78,
                'Recall': 0.81,
                'F1-Score': 0.77 },

                {'Accuracy': 0.22,
                'Precision': 0.12,
                'Recall': 0.15,
                'F1-Score': 0.19 }
            ]
                
        Returns:
            self.scores (dict): A list of scores that contains accuracy, precision, recall, and F1-score
            of between the list of :ref:`binary_mask` objects in both ``true_sources_mask_list`` 
            and ``estimated_sources_mask_list``.

        """
        scores = []
        for j in range(references.shape[-1]):
            ch_metrics = {
                'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []
            }
            for ch in range(references.shape[-2]):
                ch_metrics['Accuracy'].append(sklearn.metrics.accuracy_score(
                    references[..., ch, j], estimates[..., ch, j]
                ))
                ch_metrics['Precision'].append(sklearn.metrics.precision_score(
                    references[..., ch, j], estimates[..., ch, j]
                ))
                ch_metrics['Recall'].append(sklearn.metrics.recall_score(
                    references[..., ch, j], estimates[..., ch, j]
                ))
                ch_metrics['F1-Score'].append(sklearn.metrics.f1_score(
                    references[..., ch, j], estimates[..., ch, j]
                ))
            scores.append(ch_metrics)

        return scores
