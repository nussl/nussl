import numpy as np

from .. import MaskSeparationBase


class HighLowPassFilter(MaskSeparationBase):
    """
    Implements a super simple separation algorithm that just masks everything below
    the specified hz. It does this by zeroing out the associated FFT bins via a mask to
    produce the "high" source, and the residual is the "low" source.
    
    Args:
        input_audio_signal (AudioSignal): Signal to separate.
        high_pass_cutoff_hz (float): Cutoff in Hz. Will be rounded off 
        mask_type (str, optional): Mask type. Defaults to 'binary'.
    """

    def __init__(self, input_audio_signal, high_pass_cutoff_hz, mask_type='binary'):
        super().__init__(input_audio_signal=input_audio_signal, mask_type=mask_type)
        self.high_pass_cutoff_hz = high_pass_cutoff_hz

    def run(self):
        # Compute the spectrogram and find the closest frequency bin to the cutoff freq
        closest_freq_bin = (
            np.abs(self.audio_signal.freq_vector - self.high_pass_cutoff_hz)
        ).argmin()

        # Make masks
        low_pass_mask = self.ones_mask(self.stft.shape)
        low_pass_mask.mask[closest_freq_bin:, ...] = 0
        high_pass_mask = low_pass_mask.invert_mask()
        self.result_masks = [low_pass_mask, high_pass_mask]

        return self.result_masks
