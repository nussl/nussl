#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import copy


import spectral_utils
import separation_base
import constants
from audio_signal import AudioSignal
import utils

class RepetSim(separation_base.SeparationBase):
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
        :ref:`The REPET Demo Example <repet_demo>`
    """

    def __init__(self, input_audio_signal, similarity_threshold=None, min_distance_between_frames=None,
                 max_repeating_frames=None, high_pass_cutoff=None, do_mono=False):
        super(RepetSim, self).__init__(input_audio_signal=input_audio_signal)

        self.high_pass_cutoff = 100 if high_pass_cutoff is None else high_pass_cutoff
        self.similarity_threshold = 0 if similarity_threshold is None else similarity_threshold
        self.min_distance_between_frames = 1 if min_distance_between_frames is None else min_distance_between_frames
        self.max_repeating_frames = 100 if max_repeating_frames is None else max_repeating_frames

        self.verbose = False
        self.similarity_matrix = None
        self.bkgd = None
        self.fgnd = None
        self.similarity_indices = None
        self.magnitude_spectrogram = None

        self.do_mono = do_mono

        if self.do_mono:
            self.audio_signal = copy.copy(input_audio_signal)
            self.audio_signal.to_mono(overwrite=True)

    def run(self):

        # unpack window parameters
        win_len, win_type, hop_len, n_fft = self.stft_params.window_length, self.stft_params.window_type, \
                                           self.stft_params.hop_length, self.stft_params.n_fft_bins

        # High pass filter cutoff freq. (in # of freq. bins), +1 to match MATLAB implementation
        self.high_pass_cutoff = int(np.ceil(float(self.high_pass_cutoff) * (n_fft - 1)
                                            / self.audio_signal.sample_rate) + 1)
        self._compute_spectrum()
        self.similarity_indices = self._get_similarity_indices()

        # separate the mixture background by masking
        bkgd = np.zeros_like(self.audio_signal.audio_data)
        for i in range(self.audio_signal.num_channels):
            repeating_mask = self._compute_mask(self.magnitude_spectrogram[:, :, i])
            repeating_mask[1:self.high_pass_cutoff, :] = 1  # high-pass filter the foreground
            repeating_mask = np.vstack((repeating_mask, repeating_mask[-2:0:-1, :].conj()))
            stft_with_mask = repeating_mask * self.stft[:, :, i]
            y = spectral_utils.e_istft(stft_with_mask, win_len, hop_len, win_type,
                                       reconstruct_reflection=False, remove_padding=False)
            bkgd[i,] = y[:self.audio_signal.signal_length]

        self.bkgd = AudioSignal(audio_data_array=bkgd, sample_rate=self.audio_signal.sample_rate)

    def _compute_spectrum(self):
        self.stft = self.audio_signal.stft(overwrite=True, remove_reflection=False, use_librosa=False)

        self.magnitude_spectrogram = np.abs(self.stft[0:self.stft_params.window_length//2 + 1, :, :])

        mg = self.audio_signal.stft(overwrite=False, remove_reflection=True, use_librosa=False)

        i = 0

    def _get_similarity_indices(self):
        # unpack window overlap
        ovp = self.stft_params.window_overlap

        if self.magnitude_spectrogram is None:
            self._compute_spectrum()

        mean_magnitude_spectrogram = np.mean(self.magnitude_spectrogram, axis=2)
        self.similarity_matrix = self.compute_similarity_matrix(mean_magnitude_spectrogram.T)

        self.min_distance_between_frames *= self.audio_signal.sample_rate / ovp
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

        # Code below is adopted from http://stackoverflow.com/a/20687984/5768001

        # base similarity matrix (all dot products)
        similarity = np.dot(matrix, matrix.T)

        # inverse of the squared magnitude of preference vectors (number of occurrences)
        inv_square_mag = 1 / np.diag(similarity)

        # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
        inv_square_mag[np.isinf(inv_square_mag)] = 0

        # inverse of the magnitude
        inv_mag = np.sqrt(inv_square_mag)

        # cosine similarity (element-wise multiply by inverse magnitudes)
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
            cur_indices = utils.find_peak_indices(self.similarity_matrix[i, :],
                                                  self.max_repeating_frames,
                                                  min_dist=self.min_distance_between_frames,
                                                  threshold=self.similarity_threshold)
            cur_indices = cur_indices[:self.max_repeating_frames]
            similarity_indices.append(cur_indices)

        return similarity_indices

    def _compute_mask(self, magnitude_spectrogram):
        """

        Args:
            magnitude_spectrogram:

        Returns:

        """
        if self.magnitude_spectrogram is None:
            self._compute_spectrum()

        if self.similarity_indices is None:
            self._get_similarity_indices()

        mask = np.zeros_like(self.magnitude_spectrogram)

        for i in range(self.audio_signal.stft_length):
            cur_similarities = self.similarity_indices[i]
            similar_times = np.array([self.magnitude_spectrogram[j, :] for j in cur_similarities])
            mask[i, :] = np.median(similar_times, axis=2)

        mask = np.minimum(mask, self.magnitude_spectrogram)
        mask = (mask + constants.EPSILON) / (self.magnitude_spectrogram + constants.EPSILON)
        return mask

    @staticmethod
    def compute_repeating_mask_sim(V, I):
        """Computes the soft mask for the repeating part using the magnitude spectrogram and the similarity indices

        Parameters:
            V (np.array): 2D matrix containing the magnitude spectrogram of a signal (Lf by Lt)
            I (np.array): array containing similarity indices for all time frames
        Returns:
            M (np.array): 2D matrix (Lf by Lt) containing the soft mask for the repeating part. Elements of M take on
            values in [0,1]
         """

        Lf, Lt = np.shape(V)
        W = np.zeros_like(V).T
        for i in range(0, Lt):
            pind = I[i, :]
            W[i, :] = np.median(V.T[pind.astype(int), :], axis=0)

        W = W.T
        Wrow = np.reshape(W, (1, Lf * Lt))
        Vrow = np.reshape(V, (1, Lf * Lt))
        W = np.min(np.vstack([Wrow, Vrow]), axis=0)
        W = np.reshape(W, (Lf, Lt))
        M = (W + constants.EPSILON) / (V + constants.EPSILON)

        return M

    def get_similarity_matrix(self):
        """Calculates and returns the similarity matrix for the audio file associated with this object

        Returns:
             similarity_matrix (np.array): similarity matrix for the audio file.

        EXAMPLE::

            # Set up audio signal
            signal = nussl.AudioSignal('path_to_file.wav')

            # Set up a Repet object
            repet = nussl.Repet(signal)

            # I don't have to run repet to get a similarity matrix for signal
            sim_mat = repet.get_similarity_matrix()

        """
        self._compute_spectrum()
        V = np.mean(self.magnitude_spectrogram, axis=2)
        self.similarity_matrix = self.compute_similarity_matrix(V)
        return self.similarity_matrix

    def plot(self, output_file, **kwargs):
        import matplotlib.pyplot as plt
        plt.close('all')

        title = None

        if len(kwargs) != 0:
            if kwargs.has_key('title'):
                title = kwargs['title']

        plt.pcolormesh(self.get_similarity_matrix())
        title = title if title is not None else 'Similarity Matrix for {}'.format(self.audio_signal.file_name)
        plt.title(title)

        plt.axis('tight')
        plt.savefig(output_file)