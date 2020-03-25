import numpy as np
import librosa

from .. import MaskSeparationBase


class HPSS(MaskSeparationBase):
    """
    Implements harmonic/percussive source separation based on [1]. This is a 
    wrapper around the librosa implementation.

    References:
    
    [1] Fitzgerald, Derry. “Harmonic/percussive separation using median filtering.” 
        13th International Conference on Digital Audio Effects (DAFX10), Graz, 
        Austria, 2010.
    
    [2] Driedger, Müller, Disch. “Extending harmonic-percussive separation of audio.” 
        15th International Society for Music Information Retrieval Conference 
        (ISMIR 2014) Taipei, Taiwan, 2014.

    Parameters:
        input_audio_signal (AudioSignal): signal to separate.

        kernel_size (int or tuple (kernel_harmonic, kernel_percussive)): 
          kernel size(s) for the median filters.

        mask_type (str, optional): Mask type. Defaults to 'soft'.

        mask_threshold (float, optional): Masking threshold. Defaults to 0.5.

    """
    def __init__(self, input_audio_signal, kernel_size=31, mask_type='soft',
                 mask_threshold=0.5):
        super().__init__(
            input_audio_signal=input_audio_signal, 
            mask_type=mask_type,
            mask_threshold=mask_threshold
        )

        self.kernel_size = kernel_size

    def run(self):
        # separate the mixture background by masking
        harmonic_masks = []
        percussive_masks = []

        for ch in range(self.audio_signal.num_channels):
            # apply mask
            harmonic_mask, percussive_mask = librosa.decompose.hpss(
                self.stft[:, :, ch], kernel_size=self.kernel_size, mask=True)
            harmonic_masks.append(harmonic_mask)
            percussive_masks.append(percussive_mask)

        # make a mask and return
        harmonic_masks = np.stack(harmonic_masks, axis=-1)
        percussive_masks = np.stack(percussive_masks, axis=-1)

        _masks = np.stack([harmonic_masks, percussive_masks], axis=-1)
        
        self.result_masks = []
        
        for i in range(_masks.shape[-1]):
            mask_data = _masks[..., i]
            if self.mask_type == self.MASKS['binary']:
                mask_data = _masks[..., i] == np.max(_masks, axis=-1)
            mask = self.mask_type(mask_data)
            self.result_masks.append(mask)

        return self.result_masks
