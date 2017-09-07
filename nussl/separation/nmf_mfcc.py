#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cluster import KMeans
import librosa
import nussl.audio_signal
import nussl.constants
import nussl.spectral_utils
import nussl.utils
from nussl.transformers import transformer_nmf
from nussl.separation import separation_base
import mask_separation_base
import masks


class NMF_MFCC(mask_separation_base.MaskSeparationBase):
    """
        NMF MFCC (Non Negative Matrix Factorization using Mel Frequency Cepstral Coefficients) is a source separation
        algorithm that runs Transformer NMF on the magnitude spectrogram of an input audio signal.
        It uses K means clustering to cluster the templates and activations returned by the NMF. The dot product of the
        clustered templates and activations result in a magnitude spectrogram only containing the separated source.
        This is used to create a Binary Mask object, which can then be applied to return a list of Audio Signal objects
        corresponding to each separated source.

        References:

        Parameters:
            input_audio_signal (np.array): a 2-row Numpy matrix containing samples of the two-channel mixture.
            num_sources (int): Number of sources to find.
            num_templates (int): Number of template vectors to used in NMF.
            distance_measure (str): The type of distance measure to use in NMF - euclidean or divergence.
            num_iterations (int): The number of iterations to go through in NMF.
            random_seed (int): The seed to use in the numpy random generator
            convert_to_mono (bool): Given a stereo signal, convert to mono

        Attributes:
            input_audio_signal (object): An Audio Signal object of the input audio signal
            clusterer (object): A KMeans object for clustering the templates and activations
            signal_stft (np.matrix): The stft data for the current
            templates_matrix (np.matrix): A Numpy matrix containing the templates matrix from running NMF on the
                                                current channel from the input signal stft data.
            activation_matrix (np.matrix): A Numpy matrix containing the activation matrix from running NMF on the
                                                current channel from the input signal stft data.
            labeled_templates (np.array): A Numpy array containing the labeled templates columns
                                               from the templates matrix for a particular source
            sources (np.array): A Numpy array containing the list of Audio Signal objects for each source
            masks (np.array): A Numpy array containing the lists of Binary Mask objects for each channel

        """
    def __init__(self, input_audio_signal, num_sources, num_templates=50, distance_measure='euclidean',
                 num_iterations=50, random_seed=0, convert_to_mono=False):
        super(NMF_MFCC, self).__init__(input_audio_signal=input_audio_signal,
                                   mask_type=mask_separation_base.MaskSeparationBase.BINARY_MASK)

        self.num_sources = num_sources
        self.num_templates = num_templates
        self.distance_measure = distance_measure
        self.num_iterations = num_iterations
        self.random_seed = random_seed
        self.convert_to_mono = convert_to_mono

        self.input_audio_signal = input_audio_signal
        self.clusterer = None
        self.signal_stft = None
        self.templates_matrix = None
        self.activation_matrix = None
        self.labeled_templates = None
        self.sources = None
        self.masks = []

        # Convert the stereo signal to mono if indicated
        if self.convert_to_mono:
            self.input_audio_signal.to_mono(overwrite=True, remove_channels=False)

        # Initialize the K Means clusterer
        self.clusterer = KMeans(n_clusters=self.num_sources)
        self.input_audio_signal.stft_params = self.stft_params
        self.signal_stft = self.input_audio_signal.stft()

    def run(self):
        """ Extracts N sources from a given mixture

            Returns:
                self.masks (np.array): A list of binary mask objects that can be used to extract the sources
        """
        for i in range(self.input_audio_signal.num_channels):
            channel_stft = self.signal_stft[:, :, i]

            # Set up NMF and run
            nmf = transformer_nmf.TransformerNMF(input_matrix=np.abs(channel_stft), num_components=self.num_templates,
                                                 seed=self.random_seed)
            nmf.should_use_epsilon = False
            nmf.max_num_iterations = self.num_iterations
            nmf.distance_measure = self.distance_measure
            channel_activation_matrix, channel_templates_matrix = nmf.transform()

            # Cluster the templates matrix into Mel frequencies and retrieve labels
            cluster_templates = librosa.feature.mfcc(S=channel_templates_matrix)[1:14]
            self.clusterer.fit_transform(cluster_templates.T)
            self.labeled_templates = self.clusterer.labels_

            # Extract sources from signal
            channel_masks = self._extract_masks(channel_stft, channel_templates_matrix, channel_activation_matrix)
            self.masks.append(channel_masks)

        return self.masks

    def _extract_masks(self, signal_stft, templates_matrix, activation_matrix):
        """ Creates binary masks from clustered templates and activation matrices
        Parameters:
            signal_stft (np.matrix): A 2D Numpy matrix containing the stft of the current channel
            templates_matrix (np.matrix): A 2D Numpy matrix containing the templates matrix after running NMF on
                                          the current channel
            activation_matrix (np.matrix): A 2D Numpy matrix containing the activation matrix after running NMF on
                                          the current channel

        Returns:
            channel_mask_list (np.matrix): A list of Binary Mask objects corresponding to each source
        """

        if signal_stft is None:
            raise ValueError('Cannot extract masks with no signal_stft data')

        self.sources = []
        channel_mask_list = []
        for source_index in range(self.num_sources):
            source_indices = np.where(self.labeled_templates == source_index)[0]
            templates_mask = np.copy(templates_matrix)
            activation_mask = np.copy(activation_matrix)

            # Zero out everything but the source determined from the clusterer
            for i in range(templates_mask.shape[1]):
                templates_mask[:, i] = 0 if i in source_indices else templates_matrix[:, i]
                activation_mask[i, :] = 0 if i in source_indices else activation_matrix[i, :]

            mask_matrix = templates_mask.dot(activation_mask)
            music_stft_max = np.maximum(mask_matrix, np.abs(signal_stft))
            mask_matrix = np.divide(mask_matrix, music_stft_max)
            mask = np.nan_to_num(mask_matrix)
            mask = np.round(mask)
            mask_object = masks.BinaryMask(np.array(mask))
            channel_mask_list.append(mask_object)
        return channel_mask_list

    def make_audio_signals(self):
        """ Applies each mask in self.masks and returns a list of audio_signal objects for each source.

        Returns:
            self.sources (np.array): An array of audio_signal objects containing each separated source
        """
        for i in range(self.input_audio_signal.num_channels):
            channel_mask = self.masks[i]
            for j in range(self.num_sources):
                source = self.audio_signal.make_copy_with_stft_data((self.signal_stft[:, :, i]), verbose=False)
                source = source.apply_mask(channel_mask[j])
                source.stft_params = self.stft_params
                source.istft(overwrite=True, truncate_to_length=self.audio_signal.signal_length)
                self.sources.append(source)
        return self.sources
