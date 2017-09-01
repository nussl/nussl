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
    def __init__(self, input_audio_signal, num_sources, num_templates, distance_measure, num_iterations):
        super(NMF_MFCC, self).__init__(input_audio_signal=input_audio_signal,
                                   mask_type=mask_separation_base.MaskSeparationBase.BINARY_MASK)

        self.num_sources = num_sources
        self.num_templates = num_templates
        self.distance_measure = distance_measure
        self.num_iterations = num_iterations
        self.input_audio_signal = input_audio_signal
        self.audio_data = input_audio_signal.audio_data
        self.clusterer = None
        self.signal_stft = None
        self.templates_matrix = None
        self.activation_matrix = None
        self.labeled_templates = None
        self.sources = None
        self.masks = []
        self.clusterer = KMeans(n_clusters=self.num_sources)
        self.input_audio_signal.stft_params = self.stft_params
        self.signal_stft = self.input_audio_signal.stft()

    def run(self):
        templates_stft = self.signal_stft[:, 0:self.signal_stft.shape[1]]
        templates_stft = np.squeeze(templates_stft)

        # Set up NMF and run
        nmf = transformer_nmf.TransformerNMF(np.abs(templates_stft), self.num_templates)
        nmf.should_use_epsilon = False
        nmf.max_num_iterations = self.num_iterations
        nmf.distance_measure = self.distance_measure
        self.activation_matrix, self.templates_matrix = nmf.transform()

        # Cluster the templates matrix into Mel frequencies
        cluster_templates = librosa.feature.mfcc(S=self.templates_matrix)[1:14]
        self.clusterer.fit_transform(cluster_templates.T)

        # Retrieve cluster labels for the templates
        self.labeled_templates = self.clusterer.labels_
        self.sources = []

        # Extract sources from signal
        for source_index in range(self.num_sources):
            source_indices = np.where(self.labeled_templates == source_index)[0]
            self.extract_masks(clustered_templates=self.templates_matrix[:, source_indices],
                                                    signal_stft=self.signal_stft)
        return self.masks

    def extract_masks(self, clustered_templates, signal_stft):
        """ Extracts the binary mask objects for each source.
        Parameters:
            clustered_templates (np.matrix): The columns from self.templates_matrix that correspond to the current
                                             source being extracted
            signal_stft (np.matrix): The stft of the input audio signal
        """
        new_templates_stft = np.abs(np.squeeze(signal_stft))

        # Set up and run NMF using each clustered template
        reconstruct_nmf = transformer_nmf.TransformerNMF(input_matrix=new_templates_stft, templates=clustered_templates,
                                                         num_components=clustered_templates.shape[1], should_update_template=False)
        reconstruct_nmf.should_use_epsilon = False
        reconstruct_nmf.max_num_iterations = self.num_iterations
        reconstruct_nmf.distance_measure = self.distance_measure
        activation_matrix, new_templates_matrix = reconstruct_nmf.transform()

        # Reconstruct the signal
        reconstructed_signal = new_templates_matrix.dot(activation_matrix)

        # Mask the input signal
        music_stft_max = np.maximum(reconstructed_signal, np.abs(new_templates_stft))
        mask = np.divide(reconstructed_signal, music_stft_max)
        mask = np.nan_to_num(mask)

        # Create the binary mask
        mask = np.round(mask)
        mask_object = masks.BinaryMask(np.array(mask))
        self.masks.append(mask_object)

    def make_audio_signals(self):
        """ Applies each mask in self.masks and returns a list of audio_signal objects for each source.

        Returns:
            self.sources (np.array): An array of audio_signal objects containing each separated source
        """
        for i in range(self.num_sources):
            source_audio_signal = self.audio_signal.make_copy_with_stft_data(np.squeeze(self.signal_stft), verbose=False)
            source_audio_signal = source_audio_signal.apply_mask(self.masks[i])
            source_audio_signal.stft_params = self.stft_params
            source_audio_signal.istft(overwrite=True, truncate_to_length=self.audio_signal.signal_length)
            self.sources.append(source_audio_signal)
        return self.sources

        # # template - extracted template, residual - everything that's leftover.
        # template = np.multiply(np.squeeze(signal_stft), mask)
        # residual = np.multiply(np.squeeze(signal_stft), 1 - mask)
        #
        # return template, residual