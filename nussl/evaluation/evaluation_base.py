from itertools import permutations, combinations

import numpy as np

from .. import AudioSignal
from ..core import utils


class EvaluationBase(object):
    """
    Base class for all Evaluation classes for source separation algorithms in nussl. 
    Contains common functions for all evaluation techniques. This class should not be 
    instantiated directly.
    
    Both ``true_sources_list`` and ``estimated_sources_list`` get validated 
    using the private method :func:`_verify_input_list`. If your evaluation 
    needs to verify that input is set correctly (recommended) overwrite that method 
    to add checking.
    
    Args:
        true_sources_list (list): List of objects that contain one ground truth source per object.
            In some instances (such as the :class:`BSSEval` objects) this list is filled with
            :class:`AudioSignals` but in other cases it is populated with
            :class:`MaskBase` -derived objects (i.e., either a :class:`BinaryMask` or
            :class:`SoftMask` object).
        estimated_sources_list (list): List of objects that contain source estimations from a source
            separation algorithm. List should be populated with the same type of objects and in the
            same order as :param:`true_sources_list`.
        source_labels (list): List of strings that are labels for each source to be used as keys for
            the scores. Default value is `None` and in that case labels use the file_name attribute.
            If that is also `None`, then the source labels are `Source 0`, `Source 1`, etc.
        compute_permutation (bool): Whether or not to evaluate in a permutation-invariant 
            fashion, where the estimates are permuted to match the true sources. Only the 
            best permutation according to ``best_permutation_key`` is returned to the 
            scores dict. Defaults to False.
        best_permutation_key (str): Which metric to use to decide which permutation of 
            the sources was best.
        **kwargs (dict): Any additional keyword arguments are passed on to ``evaluate_helper``.
    """

    def __init__(self, true_sources_list, estimated_sources_list, source_labels=None, 
                 compute_permutation=False, best_permutation_key=None, **kwargs):
        self.true_sources_list = self._verify_input_list(true_sources_list)
        self.estimated_sources_list = self._verify_input_list(estimated_sources_list)
        self.keys = None

        _source_labels = []
        for i, x in enumerate(self.true_sources_list):
            if isinstance(x, AudioSignal) and x.path_to_input_file:
                label = x.path_to_input_file
            else:
                label = f'source_{i}'
            _source_labels.append(label)

        if source_labels:
            for i, label in enumerate(source_labels):
                _source_labels[i] = label

        self.source_labels = _source_labels
        self.eval_args = kwargs
        self.evaluation_object_type = type(self.true_sources_list[0])
        self._scores = {}
        self.num_channels = self.true_sources_list[0].num_channels
        self.compute_permutation = compute_permutation
        self.best_permutation_key = best_permutation_key

    @staticmethod
    def _verify_input_list(audio_signal_list):
        """
        Base method for verifying a list of input objects for an :class:`EvaluationBase`-derived
        object. Override this method when creating new :class:`EvaluationBased`-derived class.
        
        By default calls :func:`nussl.utils.verify_audio_signal_list_strict`, which verifies that
        all objects in the input list are :class:`audio_signal.AudioSignal` objects with the same
        length, sample rate and have identical number of channels.
        
        Args:
            audio_signal_list (list): List of objects that contain one signal per object. In some
                instances (such as the :class:`BSSEval`) this list is filled with
                :ref:`AudioSignals` but in other cases it is populated with :class:`MaskBase`
                -derived objects (i.e., either a :class:`BinaryMask` or :class:`SoftMask`
                object). In the latter case, this method is overridden with a specific function 
                in :class:`evaluation.precision_recall_fscore.PrecisionRecallFScore`.

        Returns:
            A verified list of objects that are ready for running the evaluation method.

        """
        return utils.verify_audio_signal_list_strict(audio_signal_list)

    def preprocess(self):
        """
        Takes the objects contained in `true_sources_list` and `estimated_sources_list`
        and processes them into numpy arrays that have shape 
        (..., n_channels, n_sources).

        Returns:
            references, estimates in that order as np.ndarrays.

        Note:
            Make sure to return the preprocessed data in the order 
            (references, estimates)!
        """
        raise NotImplementedError('Must implement preprocess in subclass!')
    
    def get_candidates(self):
        """
        This gets all the possible candidates for evaluation. If `compute_permutation`
        is False, then the estimates and the references are assumed to be in the same
        order. The first N estimates will be compared to the first N references, where
        N is min(len(estimates), len(references)).

        If `compute_permutation` is True, and `len(estimates) == len(references)`, then
        every possible ordering of the estimates will be tried to match to the references.
        So if there are 3 references and 3 estimates, a total of 3! = 6 candidates will
        be generated.

        If `compute_permutation` is True and `len(estimates) > len(references)`, then
        every combination of size `len(references)` estimates will be tried as well
        as their permutations. If there are 2 references and 4 estimates, then 
        (4 choose 2) = 6 combos will be tried. For each of those pairs of 2, there will
        be 2! = 2 permutations. So a total of 12 candidates will be generated.

        Returns:
            Two lists of combinations and permutations that should be tried. Each element
            of the list contains the indices that are used to find the sources that
            are compared to each other.

        """
        num_sources = len(self.true_sources_list)
        num_estimates = len(self.estimated_sources_list)

        if not self.compute_permutation:
            return [list(range(num_estimates))], [list(range(num_estimates))]

        combos = list(combinations(range(num_estimates), num_sources))
        orderings = list(permutations(range(num_sources)))
        return combos, orderings

    def evaluate_helper(self, references, estimates, **kwargs):
        """
        This function should be implemented by each class that inherits this
        class. The function should take in a numpy array containing the references and 
        one for the estimates and compute evaluation measures between the two arrays. 
        The results should be stored in a list of dictionaries. For example, a 
        BSSEval evaluator may return a dictionary as follows, for a single estimate:

        .. code-block:: none

                           #or windows or both
            [   {          #ch0  ch1   # results for first estimate
                    "SDR": [5.6, 5.2], # metric
                    "SIR": [9.2, 8.9], # metric
                    "SAR": [4.1, 4.3]  # metric
                }, 
                ...                    # more results for other estimates
            ]

        Each metric should be a key in the dictionary pointing to a value which is a
        list. The list will contain the metrics for however the algorithm was implemented 
        (e.g. there might be two value, one for each channel in a stereo mix, or there
        might be a sequence, one for each window that was evaluated.)
        
        Args:
            references (np.ndarray): References kept in a numpy array. Should have shape
                (..., n_channels, n_sources).
            estimates (np.ndarray): Estimates kept in whatever format you want. Should have
                shape (..., n_channels, n_sources).
            kwargs (dict): Keyword arguments with any additional arguments to be used in 
                the function (e.g. window_size, hop_length).
        
        Returns:
            A list of dictionary containing the measures corresponding to each estimate
            and reference.
        """

        raise NotImplementedError('Must implement evaluate_helper in a subclass!')

    def evaluate(self):
        """
        This function encapsulates the main functionality of all evaluation classes.
        It performs the following steps, some of which must be implemented in subclasses
        of EvaluationBase.

            1. Preprocesses the data somehow into numpy arrays that get passed into your
               evaluation function.
            2. Gets all possible candidates that will be evaluated in your evaluation function.
            3. For each candidate, runs the evaluation function (must be implemented in subclass).
            4. Finds the results from the best candidate.
            5. Returns a dictionary containing those results.

        Steps 1 and 3 must be implemented by the subclass while the others are implemented
        by EvaluationBase.
        
        Returns:
            A dictionary containing the scores for each source for the best candidate.

        """
        references, estimates = self.preprocess()
        combos, orderings = self.get_candidates()
        
        best_permutation_key = self.best_permutation_key
        scores = []
        metrics = []

        for k, combo in enumerate(combos):
            _estimates = estimates[..., combo]
            for o, order in enumerate(orderings):
                _scores = self.evaluate_helper(
                    references[..., order], _estimates, **self.eval_args)

                if not best_permutation_key or best_permutation_key not in _scores[0]:
                    best_permutation_key = sorted(_scores[0].keys())[0]

                _metrics = []
                for _score in _scores:
                    _metrics.extend(_score[best_permutation_key])

                metrics.append(np.mean(_metrics))
                scores.append((combo, order, _scores))
            
        best_idx = np.argmax(metrics)
        combo, order, score = scores[best_idx]
        results = {
            'combination': combo,
            'permutation': order
        }

        for i, o in enumerate(order):
            results[self.source_labels[o]] = score[i]
        self._scores = results

        return results

    @property
    def scores(self):
        """
        A dictionary that stores all scores from the evaluation method. Gets populated when
        :func:`evaluate` gets run.

        """
        return self._scores


class AudioSignalListMismatchError(Exception):
    """
    Error class for when true_sources_list and estimated_sources_list are different
    lengths.
    """
    pass
