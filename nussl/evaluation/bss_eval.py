import numpy as np
import museval

from .evaluation_base import EvaluationBase


def scale_bss_eval(references, estimates, idx, scaling=True):
    """
    Computes SDR, SIR, SAR for references[idx] relative to the
    chosen estimates. This only works for mono audio. Each
    channel should be done independently when calling this
    function. Lovingly borrowed from Gordon Wichern and 
    Jonathan Le Roux at Mitsubishi Electric Research Labs.
    
    Args:
        references (np.ndarray): object containing the
        references data. Of shape (n_samples, n_sources).
        
        estimates (np.ndarray): object containing the
        estimates data. Of shape (n_samples, 1).

        idx (int): Which estimates to compute metrics for.

        scaling (bool, optional): Whether to use scale-invariant (True) or
        scale-dependent (False) metrics. Defaults to True.
    
    Returns:
        tuple: SDR, SIR, SAR if inputs are numpy arrays.
    """
    references_projection = references.T @ references
    source = references[..., idx]
    scale = (
        (source @ estimates) /
        references_projection[idx, idx]
        if scaling else 1
    )

    e_true = scale * source
    e_res = estimates - e_true

    signal = (e_true ** 2).sum()
    noise = (e_res ** 2).sum()

    sdr = 10 * np.log10(signal / noise)

    references_onto_residual = np.dot(references.transpose(), e_res)
    b = np.linalg.solve(references_projection, references_onto_residual)

    e_interf = np.dot(references, b)
    e_artif = e_res - e_interf

    sir = 10 * np.log10(signal / (e_interf ** 2).sum())
    sar = 10 * np.log10(signal / (e_artif ** 2).sum())
    return sdr, sir, sar


class BSSEvaluationBase(EvaluationBase):
    """
    Base class for all evaluation classes that are based on BSSEval metrics. This 
    contains some useful verification functions, preprocessing functions that are
    used in many separation-based evaluation. Specific evaluation metrics are 
    thin wrappers around this base class, basically only implementing the
    ``self.evaluate_helper`` function.
    
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
        **kwargs (dict): Any additional arguments are passed on to evaluate_helper.
    """

    def __init__(self, true_sources_list, estimated_sources_list, source_labels=None,
                 compute_permutation=False, best_permutation_key="SDR", **kwargs):
        super().__init__(true_sources_list, estimated_sources_list, source_labels=source_labels,
                         compute_permutation=compute_permutation,
                         best_permutation_key=best_permutation_key,
                         **kwargs)

    def preprocess(self):
        """
        Implements preprocess by stacking the audio_data inside each AudioSignal
        object in both self.true_sources_list and self.estimated_sources_list.
        
        Returns:
            tuple: Tuple containing reference and estimate arrays.
        """
        references = np.stack(
            [np.copy(x.audio_data.T) for x in self.true_sources_list],
            axis=-1
        )
        estimates = np.stack(
            [np.copy(x.audio_data.T) for x in self.estimated_sources_list],
            axis=-1
        )
        return references, estimates


class BSSEvalV4(BSSEvaluationBase):
    def evaluate_helper(self, references, estimates, **kwargs):
        """
        Implements evaluation using museval.metrics.bss_eval
        """
        # museval expects shape=(nsrc, nsampl, nchan)
        # we have (nsampl, nchan, nsrc)
        # so let's massage the data so it matches before feeding it in

        references = np.transpose(references, (2, 0, 1))
        estimates = np.transpose(estimates, (2, 0, 1))

        sdr, isr, sir, sar, _ = museval.metrics.bss_eval(
            references, estimates, compute_permutation=False, **kwargs)

        scores = []
        for j in range(references.shape[0]):
            score = {
                'SDR': sdr[j], 'ISR': isr[j], 'SIR': sir[j], 'SAR': sar[j],
            }
            scores.append(score)
        return scores


class BSSEvalScale(BSSEvaluationBase):
    def preprocess(self):
        """
        Scale invariant metrics expects zero-mean centered references and sources.
        """
        references, estimates = super().preprocess()
        references -= references.mean(axis=0)
        estimates -= estimates.mean(axis=0)
        return references, estimates

    def evaluate_helper(self, references, estimates, scaling=True, **kwargs):
        """
        Implements evaluation using scale-invariant BSSEval metrics [1].

        [1] Le Roux, J., Wisdom, S., Erdogan, H., & Hershey, J. R. 
        (2019, May). SDR–half-baked or well done?. In ICASSP 2019-2019 IEEE 
        International Conference on Acoustics, Speech and Signal 
        Processing (ICASSP) (pp. 626-630). IEEE.
        """

        sdr, sir, sar = [], [], []
        for j in range(references.shape[-1]):
            cSDR, cSIR, cSAR = [], [], []
            for ch in range(references.shape[-2]):
                _SDR, _SIR, _SAR = scale_bss_eval(
                    references[..., ch, :], estimates[..., ch, j],
                    j, scaling=scaling
                )
                cSDR.append(_SDR)
                cSIR.append(_SIR)
                cSAR.append(_SAR)
            sdr.append(cSDR)
            sir.append(cSIR)
            sar.append(cSAR)

        scores = []
        for j in range(references.shape[-1]):
            score = {
                'SDR': sdr[j], 'SIR': sir[j], 'SAR': sar[j],
            }
            scores.append(score)
        return scores
