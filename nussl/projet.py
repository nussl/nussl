#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Projet for multicue separation
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

import spectral_utils
import separation_base
import constants
from audio_signal import AudioSignal
from scipy.ndimage.filters import maximum_filter, minimum_filter


class Projet(separation_base.SeparationBase):
    """Implements foreground/background separation using the 2D Fourier Transform

    Parameters:
        input_audio_signal: (AudioSignal object) The AudioSignal object that has the
                            audio data that REPET will be run on.
        use_librosa_stft: (Optional) (bool) Calls librosa's stft function instead of nussl's

    """
    def __init__(self, input_audio_signal, use_librosa_stft=constants.USE_LIBROSA_STFT, num_sources=None,
                 num_iterations=None, num_panning_directions=None, num_projections=None, verbose=None):
        super(Projet, self).__init__(input_audio_signal=input_audio_signal)
        self.sources = None
        self.use_librosa_stft = use_librosa_stft
        self.stft = None
        self.num_sources = 6 if num_sources is None else num_sources
        self.num_iterations = 200 if num_iterations is None else num_iterations
        self.num_panning_directions = 41 if num_panning_directions is None else num_panning_directions
        self.num_projections = 15 if num_projections is None else num_projections
        self.verbose = False if verbose is None else verbose
        if self.audio_signal.num_channels == 1:
            raise ValueError('Cannot run PROJET on a mono audio signal!')
        

    def run(self):
        """

        Returns:
            sources (list of AudioSignals): A list of AudioSignal objects with all of the sources found in the mixture

        Example:
             ::

        """
        self._compute_spectrum()
        (F, T, I) = self.stft.shape
        num_sources = self.num_sources
        num_possible_panning_directions = self.num_panning_directions
        num_projections = self.num_projections
        eps = 1e-20
        # initialize PSD and panning to random
        P = np.abs(np.random.randn(F * T, num_sources), dtype='float32') + 1
        Q = np.abs(np.random.randn(num_possible_panning_directions, num_sources), dtype='float32') + 1


        # compute panning profiles
        # 30 for regular gridding, the others as random
        panning_matrix = np.concatenate((self.complex_randn((I, num_possible_panning_directions - 30)),
                                         self.multichannelGrid(I, 30)), axis=1)
        panning_matrix /= np.sqrt(np.sum(np.abs(panning_matrix) ** 2, axis=0))[None, ...]

        # compute projection matrix
        # 5 for orthoganal to a regular gridding, the others as random
        projection_matrix = np.concatenate((self.complex_randn((max(num_projections - 5, 0), I)),
                                            self.orthMatrix(self.multichannelGrid(I, min(num_projections, 5)))))
        projection_matrix /= np.sqrt(np.sum(np.abs(projection_matrix) ** 2, axis=1))[..., None]

        # compute K matrix
        K = np.abs(np.dot(projection_matrix, panning_matrix)).astype(np.float32)

        # compute the projections and store their spectrograms and squared spectrograms
        C = np.reshape(np.tensordot(self.stft, projection_matrix, axes=(2, 1)), (F * T, num_projections))
        V = np.abs(C).astype(np.float32)
        V2 = V ** 2
        C = []  # release memory

        # main iterations
        for iteration in range(self.num_iterations):
            if self.verbose:
                print 'Iteration %d' % iteration
            sigma = np.dot(P, np.dot(Q.T, K.T))
            P *= np.dot(1.0 / (sigma + eps), np.dot(K, Q)) / (np.dot(3 * sigma / (sigma ** 2 + V2 + eps), np.dot(K, Q)))

            # the following line is an optional trick that enforces orthogonality of the spectrograms.
            # P*=(100+P)/(100+np.sum(P,axis=1)[...,None])

            sigma = np.dot(P, np.dot(Q.T, K.T)).T
            Q *= np.dot(K.T, np.dot(1.0 / (sigma + eps), P)) / np.dot(K.T, np.dot(3 * sigma / (sigma ** 2 + V2.T + eps), P))

        # final separation
        recompose_matrix = np.linalg.pinv(projection_matrix)  # IxM

        sigma = np.dot(P, np.dot(Q.T, K.T))
        C = np.dot(np.reshape(self.stft, (F * T, I)), projection_matrix.T)

        self.sources = []

        for j in range(num_sources):
            sigma_j = np.outer(P[:, j], np.dot(Q[:, j].T, K.T))
            source_stft = sigma_j / sigma * C
            source_stft = np.dot(source_stft, recompose_matrix.T)
            source_stft = np.reshape(source_stft, (F, T, I))
            source = AudioSignal(stft = source_stft, sample_rate = self.audio_signal.sample_rate)
            source.istft(self.stft_params.window_length, self.stft_params.hop_length, 
                        self.stft_params.window_type, overwrite=True, 
                        use_librosa=self.use_librosa_stft, 
                        truncate_to_length=self.audio_signal.signal_length)
            self.sources.append(source)
        
        return self.sources
    
    def _compute_spectrum(self):
        self.stft = self.audio_signal.stft(overwrite=True, remove_reflection=True, use_librosa=self.use_librosa_stft)

    def multichannelGrid(self, I, L, sigma=1, normalize=True):
        pos = np.linspace(0, I - 1, L)
        res = np.zeros((I, L))
        for i in range(I):
            res[i, ...] = np.exp(-(pos - i) ** 2 / sigma ** 2)
        if normalize:
            res /= np.sqrt(np.sum(res ** 2, axis=0))
        return res

    def complex_randn(self, shape):
        return np.random.randn(*shape) + 1j * np.random.randn(*shape)


    def orthMatrix(self, R):
        (I, L) = R.shape
        res = np.ones((L, I))
        res[:, -1] = - (R[-1, :] / np.squeeze(np.sum(R[:-1, :], axis=0))).T
        res /= np.sqrt(np.sum(res ** 2, axis=1))[..., None]
        return res
    
    def make_audio_signals(self):
        """ Returns the background and foreground audio signals. You must have run FT2D.run() prior
        to calling this function. This function will return None if run() has not been called.

        Returns:
            Audio Signals (List): 2 element list.

                * bkgd: Audio signal with the calculated background track
                * fkgd: Audio signal with the calculated foreground track

        EXAMPLE:
             ::
        """
        if self.sources is None:
            return None

        return self.sources
