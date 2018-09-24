#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The REpeating Pattern Extraction Technique using the Similarity Matrix (REPET-SIM)
"""

import warnings

import numpy as np

import mask_separation_base
import masks
from ..core import utils
from ..core import constants


class RepetSim(mask_separation_base.MaskSeparationBase):
    """Implements the REpeating Pattern Extraction Technique algorithm using the Similarity Matrix (REPET-SIM).

    REPET is a simple method for separating the repeating background from the non-repeating foreground in a piece of
    audio mixture. REPET-SIM is a generalization of REPET, which looks for similarities instead of periodicities.

    References:

        * Zafar Rafii and Bryan Pardo. "Audio Separation System and Method," US20130064379 A1, US 13/612,413, March 14,
          2013
        * Zafar Rafii and Bryan Pardo. "Music/Voice Separation using the Similarity Matrix," 13th International Society
          on Music Information Retrieval, Porto, Portugal, October 8-12, 2012.

    See Also:
        http://music.eecs.northwestern.edu/research.php?project=repet

    Parameters:

    Examples:
        :ref:`The RepetSim Demo Example <repet_sim_demo>`
    """

    def __init__(self, input_audio_signal, similarity_threshold=None, min_distance_between_frames=None,
                 max_repeating_frames=None, high_pass_cutoff=None, do_mono=False,
                 use_librosa_stft=constants.USE_LIBROSA_STFT, matlab_fidelity=False,
                 mask_type=mask_separation_base.MaskSeparationBase.SOFT_MASK, mask_threshold=0.5):
        super(RepetSim, self).__init__(input_audio_signal=input_audio_signal, mask_type=mask_type,
                                       mask_threshold=mask_threshold)

        self.high_pass_cutoff = 100 if high_pass_cutoff is None else high_pass_cutoff
        self.similarity_threshold = 0 if similarity_threshold is None else similarity_threshold
        self.min_distance_between_frames = 1 if min_distance_between_frames is None else min_distance_between_frames
        self.max_repeating_frames = 100 if max_repeating_frames is None else max_repeating_frames

        self._min_distance_converted_to_hops = False

        self.verbose = False
        self.similarity_matrix = None
        self.background = None
        self.foreground = None
        self.similarity_indices = None
        self.magnitude_spectrogram = None
        self.stft = None
        self.result_masks = None
        self.use_librosa_stft = use_librosa_stft
        self.matlab_fidelity = matlab_fidelity

        if self.matlab_fidelity:
            self.use_librosa_stft = False

        self.do_mono = do_mono

        if self.do_mono:
            self.audio_signal.to_mono(overwrite=True)

    def run(self):
        """
        Runs REPET-SIM, a variant of REPET using the cosine similarity matrix to find similar
        frames to do median filtering.

        Returns:

        """
        # High pass filter cutoff freq. (in # of freq. bins), +1 to match MATLAB implementation
        self.high_pass_cutoff = int(np.ceil(float(self.high_pass_cutoff) *
                                            (self.stft_params.n_fft_bins - 1) /
                                            self.audio_signal.sample_rate) + 1)
        low = 1 if self.matlab_fidelity else 0
        self._compute_spectrograms()
        self.similarity_indices = self._get_similarity_indices()

        background_stft = []
        background_mask = []
        for i in range(self.audio_signal.num_channels):
            repeating_mask = self._compute_mask(self.magnitude_spectrogram[:, :, i])

            repeating_mask[low:self.high_pass_cutoff, :] = 1  # high-pass filter the foreground
            background_mask.append(repeating_mask)

            stft_with_mask = repeating_mask * self.stft[:, :, i]
            background_stft.append(stft_with_mask)

        # Set STFT in correct order
        background_stft = np.array(background_stft).transpose((1, 2, 0))
        self._make_background_signal(background_stft)

        # make a mask and return
        background_mask = np.array(background_mask).transpose((1, 2, 0))
        background_mask = masks.SoftMask(background_mask)
        if self.mask_type == self.BINARY_MASK:
            background_mask = background_mask.mask_to_binary(self.mask_threshold)

        self.result_masks = [background_mask, background_mask.inverse_mask()]

        return self.result_masks

    def _make_background_signal(self, background_stft):
        self.background = self.audio_signal.make_copy_with_stft_data(background_stft, verbose=False)
        self.background.istft(self.stft_params.window_length, self.stft_params.hop_length, self.stft_params.window_type,
                              overwrite=True, use_librosa=self.use_librosa_stft,
                              truncate_to_length=self.audio_signal.signal_length)

    def _compute_spectrograms(self):
        self.stft = self.audio_signal.stft(overwrite=True, remove_reflection=True, use_librosa=self.use_librosa_stft)
        self.magnitude_spectrogram = np.abs(self.stft)

    def _get_similarity_indices(self):
        if self.magnitude_spectrogram is None:
            self._compute_spectrograms()

        self.similarity_matrix = self.get_similarity_matrix()

        if not self._min_distance_converted_to_hops:
            self.min_distance_between_frames *= self.audio_signal.sample_rate / self.stft_params.window_overlap
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
        assert (type(matrix) == np.ndarray)

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
        if self.similarity_matrix is None:
            raise Exception('self.similarity_matrix cannot be None')

        similarity_indices = []
        for i in range(self.audio_signal.stft_length):
            cur_indices = utils.find_peak_indices(self.similarity_matrix[i, :], self.max_repeating_frames,
                                                  min_dist=self.min_distance_between_frames,
                                                  threshold=self.similarity_threshold)

            # the first peak is always itself so we throw it out
            # we also want only self.max_repeating_frames peaks
            # so +1 for 0-based, and +1 for the first peak we threw out
            cur_indices = cur_indices[1:self.max_repeating_frames + 2]
            similarity_indices.append(cur_indices)

        if all(not idx for idx in similarity_indices):
            raise RuntimeError('No similarity indices!')

        if any(not idx for idx in similarity_indices):
            warnings.warn('Not all indices have similarities above threshold!')

        return similarity_indices

    def _compute_mask(self, magnitude_spectrogram_channel):
        """

        Args:
            magnitude_spectrogram_channel:

        Returns:

        """
        if self.magnitude_spectrogram is None:
            self._compute_spectrograms()

        if self.similarity_indices is None:
            self._get_similarity_indices()

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
        if self.magnitude_spectrogram is None:
            self._compute_spectrograms()
        mean_magnitude_spectrogram = np.mean(self.magnitude_spectrogram, axis=2)
        return self.compute_similarity_matrix(mean_magnitude_spectrogram.T)

    def make_audio_signals(self):
        """ Returns the background and foreground audio signals. You must have run Repet.run() prior
        to calling this function. This function will return None if run() has not been called.

        Returns:
            Audio Signals (List): 2 element list.

                * bkgd: Audio signal with the calculated background track
                * fkgd: Audio signal with the calculated foreground track

        Example:
            
        .. code-block:: python
            :linenos:
            
            # set up AudioSignal object
            signal = nussl.AudioSignal('path_to_file.wav')

            # set up and run RepetSim
            repet_sim = nussl.RepetSim(signal)
            repet_sim.run()

            # get audio signals (AudioSignal objects)
            background, foreground = repet_sim.make_audio_signals()
        """
        if self.background is None:
            return None

        foreground_array = self.audio_signal.audio_data - self.background.audio_data
        self.foreground = self.audio_signal.make_copy_with_audio_data(foreground_array)
        return [self.background, self.foreground]

    def plot(self, output_file, **kwargs):
        import matplotlib.pyplot as plt
        plt.close('all')

        title = None

        if len(kwargs) != 0:
            if 'title' in kwargs:
                title = kwargs['title']

        sim_mat = self.get_similarity_matrix()
        plt.pcolormesh(sim_mat)
        title = title if title is not None else 'Similarity Matrix for {}'.format(self.audio_signal.file_name)
        plt.xlabel('Time (s)')
        plt.ylabel('Time (s)')
        plt.title(title)

        plt.axis('tight')
        plt.savefig(output_file)
