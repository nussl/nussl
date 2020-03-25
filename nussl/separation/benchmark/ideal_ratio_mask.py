from ..base import MaskSeparationBase, SeparationException
from ...datasets import transforms


class IdealRatioMask(MaskSeparationBase):
    """
    Implements an ideal ratio mask (IRM) that is computed by using the known
    ground truth performance. This is one of the upper baselines.
    
    Args:
        input_audio_signal (AudioSignal): Signal to separate.
        sources (list): List of audio signal objects that correspond to the sources.
        approach (str): Either 'psa' (phase sensitive spectrum approximation) or 'msa'
          (magnitude spectrum approximation). Generally 'psa' does better.
        mask_type (str, optional): Mask type. Defaults to 'soft'.
        mask_threshold (float, optional): Masking threshold. Defaults to 0.5. 
        kwargs (dict): Extra keyword arguments are passed to the transform classes at
          initialization.
    """

    def __init__(self, input_audio_signal, sources, approach='psa', mask_type='soft',
                 mask_threshold=.5, **kwargs):

        if isinstance(sources, list):
            sources = {i: sources[i] for i in range(len(sources))}
        elif not isinstance(sources, dict):
            raise SeparationException("sources must be a list or a dict!")

        self.sources = sources
        self.approach = approach
        if self.approach == 'psa':
            tfm = transforms.PhaseSensitiveSpectrumApproximation(**kwargs)
        elif self.approach == 'msa':
            tfm = transforms.MagnitudeSpectrumApproximation(**kwargs)
        else:
            raise SeparationException(f'Unknown approach: {self.approach}')
        self.tfm = tfm

        super().__init__(
            input_audio_signal=input_audio_signal, mask_type=mask_type,
            mask_threshold=mask_threshold)

    def run(self):
        # Set up dictionary to pass to the transform.    
        data = {
            'mix': self.audio_signal,
            'sources': self.sources
        }
            
        output = self.tfm(data)
        masks = []
        mask_data = (
            output['source_magnitudes'] / 
            (output['source_magnitudes'].sum(axis=-1, keepdims=True) + 1e-8)
        )

        for i in range(mask_data.shape[-1]):            
            mask = self.mask_type(mask_data[..., i])
            masks.append(mask)

        self.result_masks = masks
        return self.result_masks
