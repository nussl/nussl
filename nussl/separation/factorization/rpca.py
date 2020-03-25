import numpy as np

from .. import MaskSeparationBase
from ..benchmark import HighLowPassFilter


class RPCA(MaskSeparationBase):
    """
    Implements foreground/background separation using RPCA.

    Huang, Po-Sen, et al. "Singing-voice separation from monaural recordings using 
    robust principal component analysis." Acoustics, Speech and Signal Processing 
    (ICASSP), 2012 IEEE International Conference on. IEEE, 2012.

    Args:
        input_audio_signal (AudioSignal): The AudioSignal object that has the
          audio data that RPCA will be run on.
        high_pass_cutoff (float, optional): Value (in Hz) for the high pass cutoff 
          filter. Defaults to 100.
        num_iterations (int, optional): how many iterations to run RPCA for. 
          Defaults to 100.
        epsilon (float, optional): Stopping criterion for RPCA convergence. 
          Defaults to 1e-7.
        mask_type (str, optional): Type of mask to use. Defaults to 'soft'.
        mask_threshold (float, optional): Threshold for mask. Defaults to 0.5.
    """

    def __init__(self, input_audio_signal, high_pass_cutoff=100, num_iterations=100, 
                 epsilon=1e-7, mask_type='soft', mask_threshold=0.5):
        super().__init__(
            input_audio_signal=input_audio_signal, 
            mask_type=mask_type,
            mask_threshold=mask_threshold)
        self.high_pass_cutoff = high_pass_cutoff

        self.epsilon = epsilon
        self.num_iterations = num_iterations
        self.gain = 1

        self.error = None
        self.magnitude_spectrogram = None

    def run(self):
        high_low = HighLowPassFilter(self.audio_signal, self.high_pass_cutoff)
        high_pass_masks = high_low.run()

        self.magnitude_spectrogram = np.abs(self.stft)

        background_masks = []
        foreground_masks = []

        for ch in range(self.audio_signal.num_channels):
            background_mask = self._compute_rpca_mask(
                self.magnitude_spectrogram[..., ch])
            foreground_mask = 1 - background_mask
            
            background_masks.append(background_mask)
            foreground_masks.append(foreground_mask)

        background_masks = np.stack(background_masks, axis=-1)
        foreground_masks = np.stack(foreground_masks, axis=-1)

        _masks = np.stack([background_masks, foreground_masks], axis=-1)

        self.result_masks = []

        for i in range(_masks.shape[-1]):
            mask_data = _masks[..., i]
            if self.mask_type == self.MASKS['binary']:
                mask_data = _masks[..., i] == np.max(_masks, axis=-1)
            
            if i == 0:
                mask_data = np.maximum(mask_data, high_pass_masks[i].mask)
            elif i == 1:
                mask_data = np.minimum(mask_data, high_pass_masks[i].mask)
            
            mask = self.mask_type(mask_data)
            self.result_masks.append(mask)

        return self.result_masks

    def _compute_rpca_mask(self, magnitude_spectrogram):
        low_rank, sparse_matrix = self.decompose(magnitude_spectrogram)
        bg_mask = self.gain * np.abs(sparse_matrix) <= np.abs(low_rank)
        return bg_mask.astype(float)

    def decompose(self, magnitude_spectrogram):
        # compute rule of thumb values of lagrange multiplier and svd-threshold
        _lambda = 1 / np.sqrt(np.max(magnitude_spectrogram.shape))

        # initialize low rank and sparse matrices to all zeros
        low_rank = np.zeros(magnitude_spectrogram.shape)
        sparse_matrix = np.zeros(magnitude_spectrogram.shape)

        # get singular values for magnitude_spectrogram
        two_norm = np.linalg.svd(magnitude_spectrogram, full_matrices=False, compute_uv=False)[0]
        inf_norm = np.linalg.norm(magnitude_spectrogram.flatten(), np.inf) / _lambda
        dual_norm = np.max([two_norm, inf_norm])
        residuals = magnitude_spectrogram / dual_norm

        # tunable parameters
        mu = 1.25 / two_norm
        mu_bar = mu * 1e7
        rho = 1.5

        error = np.inf
        converged = False
        num_iteration = 0

        while not converged and num_iteration < self.num_iterations:
            num_iteration += 1
            low_rank = self.svd_threshold(magnitude_spectrogram - sparse_matrix + residuals / mu,
                                          1 / mu)
            sparse_matrix = self.shrink(magnitude_spectrogram - low_rank + residuals / mu,
                                        _lambda / mu)
            residuals += mu * (magnitude_spectrogram - low_rank - sparse_matrix)
            mu = np.min([mu * rho, mu_bar])
            error = np.linalg.norm(magnitude_spectrogram - low_rank - sparse_matrix,
                                   ord='fro') / np.linalg.norm(magnitude_spectrogram, ord='fro')
            if error < self.epsilon:
                converged = True
        self.error = error
        return low_rank, sparse_matrix

    @staticmethod
    def shrink(matrix, tau):
        return np.sign(matrix) * np.maximum(np.abs(matrix) - tau, 0)

    def svd_threshold(self, matrix, tau):
        u, sigma, v = np.linalg.svd(matrix, full_matrices=False)
        shrunk = self.shrink(sigma, tau)
        thresholded_singular_values = np.dot(u, np.dot(np.diag(shrunk), v))
        return thresholded_singular_values
