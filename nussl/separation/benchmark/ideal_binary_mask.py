from ..base import MaskSeparationBase, SeparationException
from ...datasets import transforms


class IdealBinaryMask(MaskSeparationBase):
    """
    Implements an ideal binary mask (IBM) that is computed by using the known
    ground truth performance. This is one of the upper baselines.
    
    Args:
        input_audio_signal (AudioSignal): Signal to separate.
        sources (list): List of audio signal objects that correspond to the sources.
        mask_type (str, optional): Mask type. Defaults to 'binary'.
    """

    def __init__(self, input_audio_signal, sources, mask_type='binary',
                 mask_threshold=.5):
        if isinstance(sources, list):
            sources = {i: sources[i] for i in range(len(sources))}
        elif not isinstance(sources, dict):
            raise SeparationException("sources must be a list or a dict!")

        self.sources = sources
        super().__init__(
            input_audio_signal=input_audio_signal, mask_type=mask_type,
            mask_threshold=mask_threshold)

    def run(self):
        # Set up dictionary to pass to the transform.    
        data = {
            'mix': self.audio_signal,
            'sources': self.sources
        }

        msa = transforms.MagnitudeSpectrumApproximation()
        ibm = msa(data)['ideal_binary_mask']

        masks = []

        for i in range(ibm.shape[-1]):
            mask = self.mask_type(ibm[..., i])
            masks.append(mask)

        self.result_masks = masks
        return self.result_masks
