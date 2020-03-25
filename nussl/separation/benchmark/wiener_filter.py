import numpy as np
import norbert

from ..base import MaskSeparationBase, SeparationException


class WienerFilter(MaskSeparationBase):
    """
    Implements a multichannel Wiener filter that is computed by using some
    source estimates. When using the estimates produced by IdealRatioMask or 
    IdealBinaryMask, this is one of the upper baselines.
    
    Args:
        input_audio_signal (AudioSignal): Signal to separate.
        estimates (list): List of audio signal objects that correspond to the estimates.
        iterations (int): Number of iterations for expectation-maximization in Wiener 
          filter.
        mask_type (str, optional): Mask type. Defaults to 'soft'.
        mask_threshold (float, optional): Threshold for masking binary. Defaults to 0.5.
        kwargs (dict): Additional keyword arguments to `norbert.wiener`.
    """

    def __init__(self, input_audio_signal, estimates, iterations=1, mask_type='soft',  
                 mask_threshold=.5, **kwargs):
        if not isinstance(estimates, list):
            raise SeparationException("estimates must be a list!")

        self.estimates = estimates
        self.iterations = iterations
        self.kwargs = kwargs

        super().__init__(
            input_audio_signal=input_audio_signal, 
            mask_type=mask_type,
            mask_threshold=mask_threshold)

    def run(self):
        source_magnitudes = np.stack([
            np.abs(e.stft()) for e in self.estimates], axis=-1)
        source_magnitudes = np.transpose(source_magnitudes, (1, 0, 2, 3))
        mix_stft = np.transpose(self.audio_signal.stft(), (1, 0, 2))

        enhanced = norbert.wiener(
            source_magnitudes, mix_stft, 
            iterations=self.iterations, **self.kwargs)
        _masks = np.abs(enhanced) / np.maximum(1e-7, np.abs(mix_stft[..., None]))
        _masks = np.transpose(_masks, (1, 0, 2, 3))

        self.result_masks = []

        for i in range(_masks.shape[-1]):
            mask_data = _masks[..., i]
            if self.mask_type == self.MASKS['binary']:
                mask_data = _masks[..., i] == np.max(_masks, axis=-1)            
            mask = self.mask_type(mask_data)
            self.result_masks.append(mask)
        
        return self.result_masks
