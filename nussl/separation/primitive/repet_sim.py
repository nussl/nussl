import numpy as np

from .. import MaskSeparationBase
from ..benchmark import HighLowPassFilter
from ...core import utils
from ...core import constants


class RepetSim(MaskSeparationBase):
    """
    Implements the REpeating Pattern Extraction Technique algorithm using 
    the Similarity Matrix (REPET-SIM).

    REPET is a simple method for separating the repeating background from the 
    non-repeating foreground in a piece of audio mixture. REPET-SIM is a generalization 
    of REPET, which looks for similarities instead of periodicities.

    References:

    [1] Zafar Rafii and Bryan Pardo. 
        "Music/Voice Separation using the Similarity Matrix," 
        13th International Society on Music Information Retrieval, 
        Porto, Portugal, October 8-12, 2012.

    Args:
        input_audio_signal (AudioSignal): Audio signal to be separated.
        similarity_threshold (int, optional): Threshold for considering two
          frames to be similar. Defaults to 0.
        min_distance_between_frames (float, optional): Number of seconds two frames
          must be apart to be considered neighbors. Defaults to 1.
        max_repeating_frames (int, optional): Max number of frames to consider as
          neighbors. Defaults to 100.
        high_pass_cutoff (float, optional): Cutoff for high pass filters. Bins below
          this cutoff will be given to the background. Defaults to 100.
        mask_type (str, optional): Mask type to use.. Defaults to 'soft'.
        mask_threshold (float, optional): Threshold for mask converting to binary. 
          Defaults to 0.5.
    """

    def __init__(self, input_audio_signal, similarity_threshold=0, 
                 min_distance_between_frames=1, max_repeating_frames=100, 
                 high_pass_cutoff=100, mask_type='soft', mask_threshold=0.5):
        super().__init__(
            input_audio_signal=input_audio_signal, 
            mask_type=mask_type,
            mask_threshold=mask_threshold)

        self.high_pass_cutoff = high_pass_cutoff
        self.similarity_threshold = similarity_threshold
        self.min_distance_between_frames = min_distance_between_frames
        self.max_repeating_frames = max_repeating_frames

        self._min_distance_converted_to_hops = False

        self.similarity_matrix = None
        self.similarity_indices = None
        self.magnitude_spectrogram = None

    def run(self):
        high_low = HighLowPassFilter(self.audio_signal, self.high_pass_cutoff)
        high_pass_masks = high_low.run()

        self.magnitude_spectrogram = np.abs(self.stft)

        background_masks = []
        foreground_masks = []

        self.similarity_indices = self._get_similarity_indices()

        for ch in range(self.audio_signal.num_channels):
            background_mask = self._compute_mask(
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

    def _get_similarity_indices(self):
        self.similarity_matrix = self.get_similarity_matrix()

        if not self._min_distance_converted_to_hops:
            self.min_distance_between_frames *= (
                self.audio_signal.sample_rate / self.stft_params.hop_length
            )
            self._min_distance_converted_to_hops = True

        return self._find_similarity_indices()

    @staticmethod
    def compute_similarity_matrix(matrix):
        """Computes the cosine similarity matrix using the cosine similarity for any given input matrix X.

        Parameters:
            matrix (np.array): 2D matrix containing the magnitude spectrogram of the audio signal
        Returns:
            S (np.array): 2D similarity matrix
        """
        # ignore the 'divide by zero' warning
        with np.errstate(divide='ignore'):
            # Code below is adopted from http://stackoverflow.com/a/20687984/5768001

            # base similarity matrix (all dot products)
            similarity = np.dot(matrix, matrix.T)

            # inverse_mask of the squared magnitude of preference vectors (number of occurrences)
            inv_square_mag = 1 / np.diag(similarity)

            # if it doesn't occur, set it's inverse_mask magnitude to zero (instead of inf)
            inv_square_mag[np.isinf(inv_square_mag)] = 0

            # inverse_mask of the magnitude
            inv_mag = np.sqrt(inv_square_mag)

            # cosine similarity (element-wise multiply by inverse_mask magnitudes)
            cosine = similarity * inv_mag
            cosine = cosine.T * inv_mag

            return cosine

    def _find_similarity_indices(self):
        """Finds the similarity indices for all time frames from the similarity matrix

        Returns:
            similarity_indices (list of lists): similarity indices for all time frames
        """
        similarity_indices = []
        for i in range(self.audio_signal.stft_length):
            cur_indices = utils.find_peak_indices(
                self.similarity_matrix[i, :], self.max_repeating_frames,
                min_dist=self.min_distance_between_frames,
                threshold=self.similarity_threshold)

            # the first peak is always itself so we throw it out
            # we also want only self.max_repeating_frames peaks
            # so +1 for 0-based, and +1 for the first peak we threw out
            cur_indices = cur_indices[1:self.max_repeating_frames + 2]
            similarity_indices.append(cur_indices)

        return similarity_indices

    def _compute_mask(self, magnitude_spectrogram_channel):
        mask = np.zeros_like(magnitude_spectrogram_channel)

        for i in range(self.audio_signal.stft_length):
            cur_similarities = self.similarity_indices[i]

            if not cur_similarities:
                # If there are no similarities, then just add ones to the mask here.
                mask[:, i] = np.ones(mask.shape[constants.STFT_VERT_INDEX])
            else:
                similar_times = np.array([magnitude_spectrogram_channel[:, j]
                                          for j in cur_similarities])

                mask[:, i] = np.median(similar_times, axis=0)

        mask = np.minimum(mask, magnitude_spectrogram_channel)
        mask = (mask + constants.EPSILON) / (magnitude_spectrogram_channel + constants.EPSILON)
        return mask

    def get_similarity_matrix(self):
        """Calculates and returns the similarity matrix for the audio file associated with this object

        Returns:
             similarity_matrix (np.array): similarity matrix for the audio file.

        EXAMPLE::

            # Set up audio signal
            signal = nussl.AudioSignal('path_to_file.wav')

            # Set up a RepetSim object
            repet_sim = nussl.RepetSim(signal)

            # I don't have to run repet to get a similarity matrix for signal
            sim_mat = repet_sim.get_similarity_matrix()

        """
        mean_magnitude_spectrogram = np.mean(self.magnitude_spectrogram, axis=2)
        return self.compute_similarity_matrix(mean_magnitude_spectrogram.T)
