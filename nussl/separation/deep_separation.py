#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deep Clustering Separation Class
"""
import copy
import warnings

import torch
import librosa
import numpy as np

from ..networks import SeparationModel
from ..networks import modules
from sklearn.decomposition import PCA
from . import mask_separation_base
from . import masks


class DeepSeparation(mask_separation_base.MaskSeparationBase):
    """Implements deep source separation models using PyTorch.
    """
    def __init__(self, input_audio_signal, model_path, num_sources,
                 mask_type='soft',
                 use_librosa_stft=False,
                 clustering_options=None):

        super(DeepSeparation, self).__init__(input_audio_signal=input_audio_signal,
                                             mask_type=mask_type)
        clustering_defaults = {
            'num_clusters': num_sources,
            'n_iterations': 10,
            'covariance_type': 'tied:spherical',
            'covariance_init': 1.0
        }

        clustering_options = {**clustering_defaults, **(clustering_options if clustering_options else {})}

        self.num_sources = num_sources                                     
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.metadata = self.load_model(model_path)
        if self.audio_signal.sample_rate != self.metadata['sample_rate']:
            self.audio_signal.resample(self.metadata['sample_rate'])
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
        self.audio_signal.stft_params.n_fft_bins = self.metadata['n_fft']
        self.audio_signal.stft_params.hop_length = self.metadata['hop_length']
        self.stft = self.audio_signal.stft(overwrite=True, remove_reflection=True,
                                           use_librosa=self.use_librosa_stft)
        self.log_spectrogram = librosa.amplitude_to_db(np.abs(self.stft), ref=np.max)
        

    def _preprocess(self):
        data = {}
        data['magnitude_spectrogram'] = np.abs(self.stft)
        data['log_spectrogram'] = self.log_spectrogram.copy()
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
            
            data[key] = torch.from_numpy(data[key]).to(self.device).float()

        return data

    def run(self):
        """

        Returns:

        """
        input_data = self._preprocess()
        output = self.model(input_data)

        if 'embedding' in output:
            num_channels, sequence_length, num_features, embedding_size = output['embedding'].shape
            self.embedding = output['embedding'].data.cpu().numpy()
            self.embedding = self.embedding.reshape(-1, embedding_size)
            output['embedding'] = output['embedding'].reshape(1, num_channels*sequence_length, num_features, embedding_size)
            clusters = self.clusterer(output['embedding'])
            clusters['assignments'] = clusters['assignments'].reshape(num_channels, sequence_length, num_features, self.num_sources)
            clusters['assignments'] = clusters['assignments'].permute(3, 2, 1, 0)
            _masks = clusters['assignments'].data.cpu().numpy()
        elif 'estimates' in output:
            _masks = output['estimates']
        
        self.assignments = _masks
        self.masks = []
        for i in range(self.num_sources):
            mask = self.assignments[i, :, :, :]
            if self.mask_type == self.BINARY_MASK:
                mask = np.round(mask)
                mask_object = masks.BinaryMask(mask)
            elif self.mask_type == self.SOFT_MASK:
                mask_object = masks.SoftMask(mask)
            else:
                raise ValueError('Unknown mask type {}!'.format(self.mask_type))
            self.masks.append(mask_object)

        return self.masks

    def project_embeddings(self, num_dimensions, threshold=-80):
        """
        Does a PCA projection of the embedding space
        Args:
            num_dimensions:

        Returns:

        """
        transform = PCA(n_components=num_dimensions)
        _embedding = self.embedding[(self.log_spectrogram >= threshold).flatten()]
        output_transform = transform.fit_transform(_embedding)
        return output_transform


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

    def _repr_html_(self):
        return 'test'

    def plot(self, threshold=-25, cmap='Blues'):
        """ Plots relevant information for deep clustering onto the active figure,
            given by matplotlib.pyplot.figure()
            outside of this function. The three plots are:
                1. PCA of emeddings onto 2 dimensions for visualization
                2. The mixture mel-spectrogram.
                3. The source assignments of each tf-bin in the mixture spectrogram.
        Returns:
            None
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        import pandas as pd
        from mpl_toolkits.mplot3d import axes3d
        import matplotlib as mpl

        grid = GridSpec(6, 10)
        
        plt.subplot(grid[:3, 3:])
        plt.imshow(np.mean(self.log_spectrogram, axis=-1), origin='lower',
                   aspect='auto', cmap='magma')
        plt.xticks([])
        plt.ylabel('Frequency')
        plt.title('Mixture')

        plt.subplot(grid[:3, :3])

        output_transform = self.project_embeddings(2, threshold=threshold)
        xmin = output_transform[:, 0].min()
        xmax = output_transform[:, 0].max()
        ymin = output_transform[:, 1].min()
        ymax = output_transform[:, 1].max()

        plt.hexbin(output_transform[:, 0], output_transform[:, 1], bins=None, gridsize=100)
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xlabel('PCA dim 1')
        plt.ylabel('PCA dim 2')
        plt.title('Embedding visualization (2D)')

        ax = plt.subplot(grid[3:, :3], projection='3d')
        output_transform = self.project_embeddings(3, threshold=max(-10, threshold / 4))
        result=pd.DataFrame(output_transform, columns=['PCA%i' % i for i in range(3)])
        ax.scatter(result['PCA0'], result['PCA1'], result['PCA2'], cmap="Set2_r", s=10)
        # make simple, bare axis lines through space:
        xAxisLine = ((min(result['PCA0']), max(result['PCA0'])), (0, 0), (0,0))
        ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
        yAxisLine = ((0, 0), (min(result['PCA1']), max(result['PCA1'])), (0,0))
        ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
        zAxisLine = ((0, 0), (0,0), (min(result['PCA2']), max(result['PCA2'])))
        ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')
        # label the axes
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title("Embedding visualization (3D)")

        plt.subplot(grid[3:, 3:])
        assignments = (np.max(np.argmax(self.assignments, axis=0), axis=-1)) + 1
        silence_mask = np.mean(self.log_spectrogram, axis=-1) > threshold
        assignments *= silence_mask


        plt.imshow(assignments,
                   origin='lower', aspect='auto', cmap=cmap)
        plt.xlabel('Time (frames)')
        plt.ylabel('Frequency')
        plt.title('Source assignments')

        norm = mpl.colors.Normalize(vmin=np.min(assignments), vmax=np.max(assignments))
        labels = [f'Source {i}' for i in range(1, self.num_sources + 1)]
        values = list(range(np.min(assignments), np.max(assignments) + 1))
        mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        _legend = [mpl.patches.Patch(color=mapper.to_rgba(0), label='Silence')]

        for value, label in zip(values, labels):
            _legend.append(mpl.patches.Patch(color=mapper.to_rgba(value), label=label))
        plt.legend(handles=_legend)