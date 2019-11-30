#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import warnings

import torch
import librosa
import numpy as np

from ..deep import SeparationModel
from ..deep import modules
from sklearn.decomposition import PCA
from .mask_separation_base import MaskSeparationBase
from . import masks
from .deep_mixin import DeepMixin


class DeepMaskEstimation(MaskSeparationBase, DeepMixin):
    """Implements deep source separation models using PyTorch"""

    def __init__(
        self,
        input_audio_signal,
        model_path,
        extra_modules=None,
        mask_type='soft',
        use_librosa_stft=False,
        use_cuda=True,
    ):

        super(DeepMaskEstimation, self).__init__(
            input_audio_signal=input_audio_signal,
            mask_type=mask_type
        )

        self.device = torch.device(
            'cuda'
            if torch.cuda.is_available() and use_cuda
            else 'cpu'
        )

        self.model, self.metadata = self.load_model(model_path, extra_modules=extra_modules)
        if input_audio_signal.sample_rate != self.metadata['sample_rate']:
            input_audio_signal.resample(self.metadata['sample_rate'])

        input_audio_signal.stft_params.window_length = self.metadata['n_fft']
        input_audio_signal.stft_params.n_fft_bins = self.metadata['n_fft']
        input_audio_signal.stft_params.hop_length = self.metadata['hop_length']

        self.use_librosa_stft = use_librosa_stft
        self._compute_spectrograms()

    def _compute_spectrograms(self):
        self.stft = self.audio_signal.stft(
            overwrite=True,
            remove_reflection=True,
            use_librosa=self.use_librosa_stft
        )
        self.log_spectrogram = librosa.amplitude_to_db(
            np.abs(self.stft),
            ref=np.max
        )

    def run(self):
        """

        Returns:

        """
        input_data = self._preprocess()
        with torch.no_grad():
            output = self.model(input_data)
            output = {k: output[k] for k in output}

            if 'estimates' not in output:
                raise ValueError("This model is not a mask estimation model!")

            _masks = (output['estimates'] / input_data['magnitude_spectrogram'].unsqueeze(-1)).squeeze(0)
            _masks = _masks.permute(3, 1, 0, 2)
            _masks = _masks.cpu().data.numpy()
        

        self.assignments = _masks
        self.num_sources = self.assignments.shape[0]
        self.masks = []
        for i in range(self.assignments.shape[-1]):
            mask = self.assignments[i, :, :, :]
            mask = masks.SoftMask(mask)
            if self.mask_type == self.BINARY_MASK:
                mask = mask.mask_to_binary(1 / len(self.num_sources))
            self.masks.append(mask)

        return self.masks

    def apply_mask(self, mask):
        """
            Applies individual mask and returns audio_signal object
        """
        source = copy.deepcopy(self.audio_signal)
        source = source.apply_mask(mask)
        source.stft_params = self.audio_signal.stft_params
        source.istft(
            overwrite=True,
            truncate_to_length=self.audio_signal.signal_length
        )

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