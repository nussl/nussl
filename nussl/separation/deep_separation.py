#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deep Clustering Separation Class
"""
import copy
import warnings

try:
    import torch
    from torch.autograd import Variable
    torch_okay = True
except ImportError:
    warnings.warn('Cannot import pytorch!')
    torch_okay = False

from sklearn.cluster import KMeans
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ..networks import SeparationModel
from sklearn.decomposition import PCA
from . import mask_separation_base
from . import masks


class DeepSeparation(mask_separation_base.MaskSeparationBase):
    """Implements deep source separation models using PyTorch.
    """
    def __init__(self, input_audio_signal,
                 mask_type=mask_separation_base.MaskSeparationBase.SOFT_MASK):

        if not torch_okay:
            raise ImportError('Cannot import pytorch! Install pytorch to continue.')

        super(DeepSeparation, self).__init__(input_audio_signal=input_audio_signal,
                                             mask_type=mask_type)
        print(SeparationModel)

    def load_model(self, model_path):
        """
        Loads the model at specified path ``model_path``
        Args:
            model_path:

        Returns:

        """
        return

    def _compute_spectrograms(self):
        self.stft = self.audio_signal.stft(overwrite=True, remove_reflection=True,
                                           use_librosa=self.use_librosa_stft)
        magnitude = np.abs(self.stft)
        self.mel_spectrogram = np.empty((self.audio_signal.num_channels,
                                         self.stft.shape[1], self.num_mels))

        for i in range(self.audio_signal.num_channels):
            self.mel_spectrogram[i, :, :] = np.dot(magnitude[:, :, i].T, self.mel_filter_bank)

        self.mel_spectrogram = 10.0 * np.log10(self.mel_spectrogram**2 + 1e-7)
        self.silence_mask = self.mel_spectrogram > self.cutoff
        self.mel_spectrogram -= np.mean(self.mel_spectrogram)
        self.mel_spectrogram /= np.std(self.mel_spectrogram) + 1e-7
        return

    def deep_clustering(self):
        """

        Returns:

        """
        return

    def _extract_masks(self, ch):

        return

    def generate_mask(self, ch, assignments):
        """
            Takes binary Mel spectrogram assignments and generates mask
        """

        return 

    def run(self):
        """

        Returns:

        """
        return

    def apply_mask(self, mask):
        """
            Applies individual mask and returns audio_signal object
        """
        source = copy.deepcopy(self.audio_signal)
        source = source.apply_mask(mask)
        source.stft_params = self.stft_params
        source.istft(overwrite=True, truncate_to_length=self.audio_signal.signal_length)

        return source

    def make_audio_signals(self):
        """ Applies each mask in self.masks and returns a list of audio_signal
         objects for each source.
        Returns:
            self.sources (np.array): An array of audio_signal objects
            containing each separated source
        """
        self.sources = []
        for mask in self.masks:
            self.sources.append(self.apply_mask(mask))

        return self.sources
