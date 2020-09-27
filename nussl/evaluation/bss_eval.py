import numpy as np
import museval

from .evaluation_base import EvaluationBase


def _scale_bss_eval(references, estimate, idx, compute_sir_sar=True):
    """
    Helper for scale_bss_eval to avoid infinite recursion loop.
    """
    source = references[..., idx]
    source_energy = (source ** 2).sum()

    alpha = (
        source @ estimate / source_energy
    )

    e_true = source
    e_res = estimate - e_true

    signal = (e_true ** 2).sum()
    noise = (e_res ** 2).sum()

    snr = 10 * np.log10(signal / noise)

    e_true = source * alpha
    e_res = estimate - e_true

    signal = (e_true ** 2).sum()
    noise = (e_res ** 2).sum()

    si_sdr = 10 * np.log10(signal / noise)

    srr = -10 * np.log10((1 - (1/alpha)) ** 2)
    sd_sdr = snr + 10 * np.log10(alpha ** 2)

    si_sir = np.nan
    si_sar = np.nan

    if compute_sir_sar:
        references_projection = references.T @ references

        references_onto_residual = np.dot(references.transpose(), e_res)
        b = np.linalg.solve(references_projection, references_onto_residual)

        e_interf = np.dot(references, b)
        e_artif = e_res - e_interf

        si_sir = 10 * np.log10(signal / (e_interf ** 2).sum())
        si_sar = 10 * np.log10(signal / (e_artif ** 2).sum())

    return si_sdr, si_sir, si_sar, sd_sdr, snr, srr


def scale_bss_eval(references, estimate, mixture, idx, 
                   compute_sir_sar=True):
    """
    Computes metrics for references[idx] relative to the
    chosen estimates. This only works for mono audio. Each
    channel should be done independently when calling this
    function. Lovingly borrowed from Gordon Wichern and 
    Jonathan Le Roux at Mitsubishi Electric Research Labs.

    This returns 9 numbers (in this order):

    - SI-SDR: Scale-invariant source-to-distortion ratio. Higher is better.
    - SI-SIR: Scale-invariant source-to-interference ratio. Higher is better.
    - SI-SAR: Scale-invariant source-to-artifact ratio. Higher is better.
    - SD-SDR: Scale-dependent source-to-distortion ratio. Higher is better.
    - SNR: Signal-to-noise ratio. Higher is better.
    - SRR: The source-to-rescaled-source ratio. This corresponds to 
      a term that punishes the estimate if its scale is off relative
      to the reference. This is an unnumbered equation in [1], but
      is the term on page 2, second column, second to last line:
      ||s - alpha*s||**2. s here is factored out. Higher is better.
    - SI-SDRi: Improvement in SI-SDR over using the mixture as the estimate.
    - SD-SDRi: Improvement in SD-SDR over using the mixture as the estimate.
    - SNRi: Improvement in SNR over using the mixture as the estimate.

    References:

    [1] Le Roux, J., Wisdom, S., Erdogan, H., & Hershey, J. R. 
        (2019, May). SDR–half-baked or well done?. In ICASSP 2019-2019 IEEE 
        International Conference on Acoustics, Speech and Signal 
        Processing (ICASSP) (pp. 626-630). IEEE.
    
    Args:
        references (np.ndarray): object containing the
          references data. Of shape (n_samples, n_sources).
         
        estimate (np.ndarray): object containing the
          estimate data. Of shape (n_samples, 1).

        mixture (np.ndarray): objct containingthe
          mixture data. Of shape (n_samples, 1).

        idx (int): Which reference to compute metrics against.

        compute_sir_sar (bool, optional): Whether or not to compute SIR/SAR
          metrics, which can be computationally expensive and may not be
          relevant for your evaluation. Defaults to True

    Returns:
        tuple: SI-SDR, SI-SIR, SI-SAR, SD-SDR, SNR, SRR, SI-SDRi, SD-SDRi, SNRi
    """    
    si_sdr, si_sir, si_sar, sd_sdr, snr, srr = _scale_bss_eval(
        references, estimate, idx, compute_sir_sar=compute_sir_sar)
    mix_metrics = _scale_bss_eval(
        references, mixture, idx, compute_sir_sar=False)

    mix_si_sdr = mix_metrics[0]
    mix_sd_sdr = mix_metrics[3]
    mix_snr = mix_metrics[4]
    si_sdri = si_sdr - mix_metrics[0]
    sd_sdri = sd_sdr - mix_metrics[3]
    snri = snr - mix_metrics[4]

    return (
      si_sdr, si_sir, si_sar, sd_sdr, snr, srr, si_sdri, sd_sdri, snri, 
      mix_si_sdr, mix_sd_sdr, mix_snr
    )


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
    # TODO: Populate score in evaluation_helper() using self.keys.
    keys = ['SDR', 'ISR', 'SIR', 'SAR']
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
            [x.audio_data for x in self.true_sources_list],
            axis=-1
        )
        estimates = np.stack(
            [x.audio_data for x in self.estimated_sources_list],
            axis=-1
        )
        return references.transpose((1, 0, 2)), estimates.transpose((1, 0, 2))


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
                'SDR': sdr[j].tolist(), 
                'ISR': isr[j].tolist(), 
                'SIR': sir[j].tolist(), 
                'SAR': sar[j].tolist(),
            }
            scores.append(score)
        return scores


class BSSEvalScale(BSSEvaluationBase):
    # TODO: Populate score in evaluation_helper() using self.keys.
    keys = [
        'SI-SDR', 'SI-SIR', 'SI-SAR',
        'SD-SDR', 'SNR', 'SRR',
        'SI-SDRi', 'SD-SDRi', 'SNRi',
        'MIX-SI-SDR', 'MIX-SD-SDR', 'MIX-SNR',
    ]
    def __init__(self, true_sources_list, estimated_sources_list, source_labels=None,
                 compute_permutation=False, best_permutation_key="SI-SDR", **kwargs):
        super().__init__(true_sources_list, estimated_sources_list, source_labels=source_labels,
                         compute_permutation=compute_permutation,
                         best_permutation_key=best_permutation_key,
                         **kwargs)

    def preprocess(self):
        """
        Scale invariant metrics expects zero-mean centered references and sources.
        """
        references, estimates = super().preprocess()

        mixture = references.sum(axis=-1)
        mixture -= mixture.mean(axis=0)

        self.mixture = mixture
        references -= references.mean(axis=0)
        estimates -= estimates.mean(axis=0)

        return references, estimates

    def evaluate_helper(self, references, estimates, compute_sir_sar=True):
        """
        Implements evaluation using new BSSEval metrics [1]. This computes every
        metric described in [1], including:

        - SI-SDR: Scale-invariant source-to-distortion ratio. Higher is better.
        - SI-SIR: Scale-invariant source-to-interference ratio. Higher is better.
        - SI-SAR: Scale-invariant source-to-artifact ratio. Higher is better.
        - SD-SDR: Scale-dependent source-to-distortion ratio. Higher is better.
        - SNR: Signal-to-noise ratio. Higher is better.
        - SRR: The source-to-rescaled-source ratio. This corresponds to 
          a term that punishes the estimate if its scale is off relative
          to the reference. This is an unnumbered equation in [1], but
          is the term on page 2, second column, second to last line:
          ||s - alpha*s||**2. s is factored out. Higher is better.
        - SI-SDRi: Improvement in SI-SDR over using the mixture as the estimate. Higher 
          is better.
        - SD-SDRi: Improvement in SD-SDR over using the mixture as the estimate. Higher
          is better.
        - SNRi: Improvement in SNR over using the mixture as the estimate. Higher is
          better.

        Note:

        If `compute_sir_sar = False`, then you'll get `np.nan` for SI-SIR and 
        SI-SAR!

        References:

        [1] Le Roux, J., Wisdom, S., Erdogan, H., & Hershey, J. R. 
        (2019, May). SDR–half-baked or well done?. In ICASSP 2019-2019 IEEE 
        International Conference on Acoustics, Speech and Signal 
        Processing (ICASSP) (pp. 626-630). IEEE.
        """

        sisdr, sisir, sisar, sdsdr, snr, srr = [], [], [], [], [], []
        sisdri, sdsdri, snri, mix_sisdr, mix_sdsdr, mix_snr = [], [], [], [], [], []
        for j in range(references.shape[-1]):
            cSISDR, cSISIR, cSISAR, cSDSDR, cSNR, cSRR = [], [], [], [], [], []
            cSISDRi, cSDSDRi, cSNRi, cMIXSISDR, cMIXSDSDR, cMIXSNR = [], [], [], [], [], []
            for ch in range(references.shape[-2]):
                output = scale_bss_eval(
                        references[..., ch, :], estimates[..., ch, j], 
                        self.mixture[..., ch], j, compute_sir_sar=compute_sir_sar
                    )
                _SISDR, _SISIR, _SISAR, _SDSDR, _SNR, _SRR, = output[:6]
                _SISDRi, _SDSDRi, _SNRi, _MIXSISDR, _MIXSDSDR, _MIXSNR = output[6:]
            
                cSISDR.append(_SISDR)
                cSISIR.append(_SISIR)
                cSISAR.append(_SISAR)
                cSDSDR.append(_SDSDR)
                cSNR.append(_SNR)
                cSRR.append(_SRR)
                cSISDRi.append(_SISDRi)
                cSDSDRi.append(_SDSDRi)
                cSNRi.append(_SNRi)
                cMIXSISDR.append(_MIXSISDR) 
                cMIXSDSDR.append(_MIXSDSDR)
                cMIXSNR.append(_MIXSNR)

            sisdr.append(cSISDR)
            sisir.append(cSISIR)
            sisar.append(cSISAR)

            sdsdr.append(cSDSDR)
            snr.append(cSNR)
            srr.append(cSRR)

            sisdri.append(cSISDRi)
            sdsdri.append(cSDSDRi)
            snri.append(cSNRi)

            mix_sisdr.append(cMIXSISDR)
            mix_sdsdr.append(cMIXSDSDR)
            mix_snr.append(cMIXSNR)

        scores = []
        for j in range(references.shape[-1]):
            score = {
                'SI-SDR': sisdr[j], 'SI-SIR': sisir[j], 'SI-SAR': sisar[j],
                'SD-SDR': sdsdr[j], 'SNR': snr[j], 'SRR': srr[j], 
                'SI-SDRi': sisdri[j], 'SD-SDRi': sdsdri[j], 'SNRi': snri[j],
                'MIX-SI-SDR': mix_sisdr[j], 'MIX-SD-SDR': mix_sdsdr[j],
                'MIX-SNR': mix_snr[j],
            }
            scores.append(score)
        return scores
