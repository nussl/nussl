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
    def __init__(self, input_audio_signal, num_sources, num_templates, distance_measure,
                 num_iterations, random_seed, mono):
        super(NMF_MFCC, self).__init__(input_audio_signal=input_audio_signal,
                                   mask_type=mask_separation_base.MaskSeparationBase.BINARY_MASK)

        self.num_sources = num_sources
        self.num_templates = num_templates
        self.distance_measure = distance_measure
        self.num_iterations = num_iterations
        self.random_seed = None
        self.mono=False

        self.input_audio_signal = input_audio_signal
        self.audio_data = input_audio_signal.audio_data
        self.clusterer = None
        self.signal_stft = None
        self.templates_matrix = None
        self.activation_matrix = None
        self.labeled_templates = None
        self.sources = None
        self.masks = []

        if self.mono:
            self.input_audio_signal.audio_data.to_mono(overwrite=True)

        self.clusterer = KMeans(n_clusters=self.num_sources)
        self.input_audio_signal.stft_params = self.stft_params
        self.signal_stft = self.input_audio_signal.stft()

    def run(self):
        for i in range(self.audio_signal.num_channels):
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
            channel_masks = self.extract_masks(channel_stft, channel_templates_matrix, channel_activation_matrix)
            self.masks.append(channel_masks)

        return self.masks

    def extract_masks(self, signal_stft, templates_matrix, activation_matrix):
        if signal_stft is None:
            raise ValueError('Cannot extract masks with no signal_stft data')

        self.sources = []
        channel_mask_list = []
        for source_index in range(self.num_sources):
            source_indices = np.where(self.labeled_templates == source_index)[0]
            templates_mask = np.copy(templates_matrix)
            activation_mask = np.copy(activation_matrix)
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
        for i in range(self.audio_signal.num_channels):
            channel_mask = self.masks[i]
            for j in range(self.num_sources):
                source_audio_signal = self.audio_signal.make_copy_with_stft_data((self.signal_stft[:,:,i]), verbose=False)
                source_audio_signal = source_audio_signal.apply_mask(channel_mask[j])
                source_audio_signal.stft_params = self.stft_params
                source_audio_signal.istft(overwrite=True, truncate_to_length=self.audio_signal.signal_length)
                self.sources.append(source_audio_signal)
        return self.sources
