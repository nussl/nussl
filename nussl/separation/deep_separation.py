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
from ..networks import modules
from sklearn.decomposition import PCA
from . import mask_separation_base
from . import masks


class DeepSeparation(mask_separation_base.MaskSeparationBase):
    """Implements deep source separation models using PyTorch.
    """
    def __init__(self, input_audio_signal, model_path, num_sources,
                 mask_type=mask_separation_base.MaskSeparationBase.SOFT_MASK,
                 use_librosa_stft=True,
                 clustering_options=None):

        if not torch_okay:
            raise ImportError('Cannot import pytorch! Install pytorch to continue.')
        super(DeepSeparation, self).__init__(input_audio_signal=input_audio_signal,
                                             mask_type=mask_type)
        clustering_defaults = {
            'num_clusters': num_sources,
            'n_iterations': 10,
            'covariance_type': 'tied:spherical',
            'covariance_init': 5.0
        }

        clustering_options = {**clustering_defaults, **(clustering_options if clustering_options else {})}

        self.num_sources = num_sources                                     
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.metadata = self.load_model(model_path)
        self.use_librosa_stft = use_librosa_stft
        self.clusterer = modules.Clusterer(**clustering_options)
        self._compute_spectrograms()

    def load_model(self, model_path):
        """
        Loads the model at specified path ``model_path``
        Args:
            model_path:

        Returns:

        """
        model_dict = torch.load(model_path)
        model = SeparationModel(model_dict['config'])
        model.load_state_dict(model_dict['state_dict'])
        model = model.to(self.device)
        metadata = model_dict['metadata'] if 'metadata' in model_dict else {}
        return model, metadata

    def _compute_spectrograms(self):
        self.audio_signal.stft_params.window_length = self.metadata['n_fft']
        self.audio_signal.stft_params.hop_length = self.metadata['hop_length']
        self.stft = self.audio_signal.stft(overwrite=True, remove_reflection=True,
                                           use_librosa=self.use_librosa_stft)
        

    def _preprocess(self):
        data = {}
        data['magnitude_spectrogram'] = np.abs(self.stft)
        data['log_spectrogram'] = librosa.amplitude_to_db(data['magnitude_spectrogram'], ref=np.max)
        data['log_spectrogram'] -= np.mean(data['log_spectrogram'])
        data['log_spectrogram'] /= np.std(data['log_spectrogram']) + 1e-7

        for key in data:
            """[num_batch, sequence_length, num_frequencies*num_channels, ...], 
        while 'cnn' produces [num_batch, num_channels, num_frequencies, sequence_length, ...]
        """
            if self.metadata['format'] == 'rnn':
                data[key] = np.expand_dims(data[key], axis=0)
                if self.metadata['num_channels'] != self.audio_signal.num_channels:
                    data[key] = np.swapaxes(data[key], 0, 3)

                _shape = data[key].shape
                shape = [_shape[0], _shape[1], _shape[2]*_shape[3]]
                data[key] = np.reshape(data[key], shape)
                data[key] = np.swapaxes(data[key], 1, 2)
                
                    
            elif self.metadata['format'] == 'cnn':
                axes_loc = [0, 3, 2, 1]
                data[key] = np.moveaxis(data[key], [0, 1, 2, 3], axes_loc)
            
            data[key] = torch.from_numpy(data[key]).to(self.device)

        return data

    def run(self):
        """

        Returns:

        """
        input_data = self._preprocess()
        output = self.model(input_data)

        if 'embedding' in output:
            num_channels, sequence_length, num_features, embedding_size = output['embedding'].shape
            output['embedding'] = output['embedding'].reshape(1, num_channels*sequence_length, num_features, embedding_size)
            clusters = self.clusterer(output['embedding'])
            clusters['assignments'] = clusters['assignments'].reshape(num_channels, sequence_length, num_features, self.num_sources)
            clusters['assignments'] = clusters['assignments'].permute(3, 2, 1, 0)
            _masks = clusters['assignments'].data.cpu().numpy()
        elif 'estimates' in output:
            _masks = output['estimates']
        
        self.masks = []
        for i in range(self.num_sources):
            mask = _masks[i, :, :, :]
            if self.mask_type == self.BINARY_MASK:
                mask = np.round(mask)
                mask_object = masks.BinaryMask(mask)
            elif self.mask_type == self.SOFT_MASK:
                mask_object = masks.SoftMask(mask)
            else:
                raise ValueError('Unknown mask type {}!'.format(self.mask_type))
            self.masks.append(mask_object)

        return self.masks


    def apply_mask(self, mask):
        """
            Applies individual mask and returns audio_signal object
        """
        source = copy.deepcopy(self.audio_signal)
        source = source.apply_mask(mask)
        source.stft_params = self.audio_signal.stft_params
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
