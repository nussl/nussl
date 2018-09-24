#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Projet for spatial audio separation
from this paper:

@inproceedings{fitzgeraldPROJETa,
TITLE = {{PROJET - Spatial Audio Separation Using Projections}},
AUTHOR = {D. Fitzgerald and A. Liutkus and R. Badeau},
BOOKTITLE = {{41st International Conference on Acoustics, Speech and Signal Processing (ICASSP)}},
ADDRESS = {Shanghai, China},
PUBLISHER = {{IEEE}},
YEAR = {2016},
}

Copyright (c) 2016, Antoine Liutkus, Inria

modified by Ethan Manilow and Prem Seetharaman for incorporation into nussl.
"""

import numpy as np

import separation_base
from ..core import utils
from ..core import constants
from ..core.audio_signal import AudioSignal


class Projet(separation_base.SeparationBase):
    """Implements foreground/background separation using the 2D Fourier Transform

    Parameters:
        input_audio_signal: (AudioSignal object) The AudioSignal object that has the
                            audio data that REPET will be run on.
        use_librosa_stft: (Optional) (bool) Calls librosa's stft function instead of nussl's

    """
    def __init__(self, input_audio_signal, num_sources,
                 num_iterations=200, num_panning_directions=41, num_projections=15,
                 matrix_datatype='float32', panning_profiles=30,
                 verbose=False, use_librosa_stft=constants.USE_LIBROSA_STFT):
        super(Projet, self).__init__(input_audio_signal=input_audio_signal)
        
        if not self.audio_signal.is_stereo:
            raise ValueError('Can only run PROJET on a stereo audio signal!')

        self.num_sources = num_sources
        self.num_iterations = num_iterations
        self.num_panning_directions = num_panning_directions
        self.num_projections = num_projections
        self.panning_profiles = panning_profiles

        if isinstance(matrix_datatype, str):
            matrix_datatype = np.dtype(matrix_datatype)

        if not np.issubdtype(matrix_datatype, np.float):
            raise ValueError('matrix_datatype must be a float!')

        self.matrix_datatype = matrix_datatype

        self.verbose = verbose

        self.stft = None
        self.sources = None
        self.use_librosa_stft = use_librosa_stft

    def run(self):
        """

        Returns:
            sources (list of AudioSignals): A list of AudioSignal objects with all of the sources found in the mixture

        Example:
             ::

        """
        self._compute_spectrograms()
        
        (num_freq_bins, num_time_bins, num_channels) = self.stft.shape
        num_sources = self.num_sources
        eps = 1e-20

        if self.verbose: print('Initializing panning matrix...')
        # initialize PSD and panning to random
        P = np.abs(np.random.randn(num_freq_bins * num_time_bins, num_sources)).astype(self.matrix_datatype) + 1

        # panning_sources_matrix is number of panning directions
        # to look for by number of sources (Q in original paper)
        panning_sources_matrix = np.abs(np.random.randn(self.num_panning_directions,
                                                        num_sources)).astype(self.matrix_datatype) + 1

        chan_pan_diff = utils.complex_randn((num_channels, self.num_panning_directions - self.panning_profiles))
        chan_per_panning_profiles = self.multichannel_grid(num_channels, self.panning_profiles)

        if self.verbose: print('Computing initial panning profiles...')

        # compute panning profiles
        panning_matrix = np.concatenate((chan_pan_diff, chan_per_panning_profiles), axis=1)
        panning_matrix /= np.sqrt(np.sum(np.abs(panning_matrix) ** 2, axis=0))[None, ...]

        if self.verbose: print('Computing initial projection matrices...')

        # compute projection matrix
        projection_matrix = np.concatenate((utils.complex_randn((max(self.num_projections - 5, 0), num_channels)),
                                            self.orthogonal_matrix(self.multichannel_grid(num_channels, min(self.num_projections, 5)))))
        projection_matrix /= np.sqrt(np.sum(np.abs(projection_matrix) ** 2, axis=1))[..., None]

        if self.verbose: print('Computing K matrix.')
        # compute K matrix
        K = np.abs(np.dot(projection_matrix, panning_matrix)).astype(np.float32)

        if self.verbose: print('Computing projections and storing spectrograms and squared spectrograms.')
        # compute the projections and store their spectrograms and squared spectrograms
        C = np.tensordot(self.stft, projection_matrix, axes=(2, 1))
        C = np.reshape(C, (num_freq_bins * num_time_bins, self.num_projections))

        # NOTE: C now the same shape as P.
        V = np.abs(C).astype(np.float32)
        V2 = V ** 2

        # noinspection PyUnusedLocal
        C = []  # release memory

        if self.verbose: print('Starting iterations')
        # main iterations
        for iteration in range(self.num_iterations):

            if self.verbose:
                print('Iteration {}'.format(iteration))

            sigma = np.dot(P, np.dot(panning_sources_matrix.T, K.T))

            if self.verbose: print('\tUpdating P...')
            # updating P
            P *= np.dot(1.0 / (sigma + eps), np.dot(K, panning_sources_matrix)) / \
                 (np.dot(3 * sigma / (sigma ** 2 + V2 + eps), np.dot(K, panning_sources_matrix)))

            if self.verbose: print('\tUpdating sigma')
            # the following line is an optional trick that enforces orthogonality of the spectrograms.
            # P*=(100+P)/(100+np.sum(P,axis=1)[...,None])
            # update sigma using updated P. transpose to fit into Q. (15, F*T)
            sigma = np.dot(P, np.dot(panning_sources_matrix.T, K.T)).T

            if self.verbose: print('\tUpdating panning sources matrix')
            # updating Q
            panning_sources_matrix *= np.dot(K.T, np.dot(np.divide(1.0, sigma + eps), P)) / \
                 np.dot(K.T, np.dot(np.divide(3 * sigma,  (sigma ** 2 + V2.T + eps)), P))

        if self.verbose: print('Completing final separation')
        # final separation
        recompose_matrix = np.linalg.pinv(projection_matrix)  # IxM

        sigma = np.dot(P, np.dot(panning_sources_matrix.T, K.T))
        C = np.dot(np.reshape(self.stft, (num_freq_bins * num_time_bins, num_channels)), projection_matrix.T)

        self.sources = []

        if self.verbose: print('Making AudioSignal objects')
        for j in range(num_sources):
            sigma_j = np.outer(P[:, j], np.dot(panning_sources_matrix[:, j].T, K.T))
            source_stft = sigma_j / sigma * C
            source_stft = np.dot(source_stft, recompose_matrix.T)
            source_stft = np.reshape(source_stft, (num_freq_bins, num_time_bins, num_channels))
            source = AudioSignal(stft=source_stft, sample_rate=self.audio_signal.sample_rate)
            source.istft(self.stft_params.window_length, self.stft_params.hop_length, 
                        self.stft_params.window_type, overwrite=True, 
                        use_librosa=self.use_librosa_stft, 
                        truncate_to_length=self.audio_signal.signal_length)
            self.sources.append(source)

        if self.verbose: print('Projet finished running.')
        return self.sources
    
    def _compute_spectrograms(self):
        self.stft = self.audio_signal.stft(overwrite=True, remove_reflection=True, use_librosa=self.use_librosa_stft)

    def multichannel_grid(self, I, L, sigma=1, normalize=True):
        # 15 points equally spaced between 0 and num_channels - 1 (1). 15 points between 0 and 1 basically.
        pos = np.linspace(0, I - 1, L)
        # 2 by 15 all 0s.
        res = np.zeros((I, L))
        for i in range(I):
            # each row becomes e^(+/-[0, 1]**2)
            res[i, ...] = np.exp(-(pos - i) ** 2 / sigma ** 2)
        if normalize:
            res /= np.sqrt(np.sum(res ** 2, axis=0))
        return res

    def orthogonal_matrix(self, R):
        # 2 by 15
        (I, L) = R.shape
        # 15 by 2
        res = np.ones((L, I))
        # all rows, squeeze removes all one dimensional entries (1, 3, 1) shape goes to (3) shape.
        # sum of all rows of R but the last one, along each column.
        # all columns of res but the last one become the last row of R (2 by 1)
        # divided by the sum of all the columns of R but the last one.
        # transpose to fit into res.
        res[:, -1] = - (R[-1, :] / np.squeeze(np.sum(R[:-1, :], axis=0))).T
        # normalize res by rms along each row
        res /= np.sqrt(np.sum(res ** 2, axis=1))[..., None]
        return res
    
    def make_audio_signals(self):
        """ Returns the background and foreground audio signals. You must have run FT2D.run() prior
        to calling this function. This function will return None if run() has not been called.

        Returns:
            Audio Signals (List):  element list.

        EXAMPLE:
             ::
        """

        return self.sources

    def plot(self, output_name, **kwargs):
        pass
