#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import warnings

import librosa
import numpy as np
from sklearn.decomposition import PCA

from ...deep import SeparationModel
from ...deep import modules
from .. import mask_separation_base
from .. import masks

from .clusterers import GaussianMixtureConfidence, KMeansConfidence, SpectralClusteringConfidence


class ClusteringSeparationBase(mask_separation_base.MaskSeparationBase):
    """Implements deep source separation models using PyTorch"""

    def __init__(
        self,
        input_audio_signal,
        num_sources,
        mask_type='soft',
        use_librosa_stft=False,
        clustering_options=None,
        alpha=1.0,
        percentile=0,
        clustering_type='kmeans'
    ):
        super(ClusteringSeparationBase, self).__init__(
            input_audio_signal=input_audio_signal,
            mask_type=mask_type
        )

        self.num_sources = num_sources
        self.clustering_options = (
            {} if clustering_options is None else clustering_options 
        )
        self.use_librosa_stft = use_librosa_stft
        
        allowed_clustering_types = ['kmeans', 'gmm', 'spectral_clustering']
        if clustering_type not in allowed_clustering_types:
            raise ValueError(
                f"clustering_type = {clustering_type} not allowed!" 
                f"Use one of {allowed_clustering_types}."
            )

        self.clustering_type = clustering_type
        self.clusterer = None
        self.percentile = percentile
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

        threshold = self.log_spectrogram
        self.threshold = (threshold > np.percentile(threshold, self.percentile))

    def init_clusterer(self):
        if self.clustering_type == 'kmeans':
            clusterer = KMeansConfidence
        elif self.clustering_type == 'gmm':
            clusterer = GaussianMixtureConfidence
        elif self.clustering_type == 'spectral_clustering':
            self.clustering_options['weights'] = (
                np.abs(self.stft.flatten())
            )
            clusterer = SpectralClusteringConfidence
        
        return clusterer(
            n_components=self.num_sources,
            **self.clustering_options
        )

    def project_data(self, data):
        return data

    def extract_features(self):
        raise NotImplementedError()

    def cluster_features(self, features, clusterer):
        threshold = self.project_data(self.threshold)
        threshold = threshold.astype(bool)

        if self.clustering_type != 'spectral_clustering':
            clusterer.fit(features[threshold.flatten()])
        else:
            clusterer.fit(features)
        assignments, confidence = clusterer.predict_and_get_confidence(features)
        return assignments, confidence

    def postprocess(self, assignments, confidence):
        assignments = assignments.reshape(self.stft.shape + (self.num_sources,))
        confidence = confidence.reshape(self.stft.shape)
        assignments = assignments.transpose(3, 0, 1, 2)
        return assignments, confidence

    def run(self):
        """

        Returns:

        """
        if self.clusterer is None:
            self.clusterer = self.init_clusterer()
        self.features = self.extract_features()
        self.assignments, self.confidence = self.cluster_features(
            self.features, self.clusterer
        )
        self.assignments, self.confidence = self.postprocess(
            self.assignments, self.confidence
        )

        self.masks = []
        for i in range(self.num_sources):
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

    def project_embeddings(self, num_dimensions, threshold=-80):
        """
        Does a PCA projection of the embedding space
        Args:
            num_dimensions (int): Number of dimensions to project down to
            threshold (float): Cutoff for plotting points

        Returns:

        """
        transform = PCA(n_components=num_dimensions)
        mask = (self.log_spectrogram >= threshold).astype(float).reshape(
            self.log_spectrogram.shape[0], -1).T
        mask = self.project_data(mask)
        mask = mask.T.reshape(-1) > 0
        _embedding = self.features[mask]
        output_transform = transform.fit_transform(_embedding)
        return output_transform

    def plot_features_1d(self, threshold, bins=150):
        import matplotlib.pyplot as plt
        output_transform = self.project_embeddings(1, threshold=threshold)
        plt.hist(output_transform, bins=bins)
        plt.xlabel('PCA dim 1')
        plt.ylabel('Count')
        plt.title('Embedding visualization (2D)')

    def plot_features_2d(self, threshold):
        import matplotlib.pyplot as plt
        output_transform = self.project_embeddings(2, threshold=threshold)
        xmin = output_transform[:, 0].min()
        xmax = output_transform[:, 0].max()
        ymin = output_transform[:, 1].min()
        ymax = output_transform[:, 1].max()

        plt.hexbin(
            output_transform[:, 0],
            output_transform[:, 1],
            bins='log',
            gridsize=100
        )
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xlabel('PCA dim 1')
        plt.ylabel('PCA dim 2')
        plt.title('Embedding visualization (2D)')

    def plot_features_3d(self, threshold, ax):
        import pandas as pd
        output_transform = self.project_embeddings(
            3,
            threshold=threshold / 4,
        )
        result=pd.DataFrame(
            output_transform,
            columns=['PCA%i' % i for i in range(3)]
        )
        ax.scatter(
            result['PCA0'],
            result['PCA1'],
            result['PCA2'],
            cmap="Set2_r",
            s=10
        )
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

    def plot(self, cmap='Blues', bins=150):
        """Plots relevant information for clustering onto the active
        figure, given by matplotlib.pyplot.figure() outside of this function.
        The four plots are:
            1. PCA of emeddings onto 2 dimensions for visualization (if possible)
            2. The mixture mel-spectrogram.
            3. PCA of embeddings onto 3 dimensions (if possible)
            4. The source assignments of each tf-bin in the mixture spectrogram.

        Returns:
            None
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from mpl_toolkits.mplot3d import axes3d
        import matplotlib as mpl

        threshold = np.percentile(
            self.log_spectrogram, 
            max(90, self.percentile)
        )

        grid_sizes = (18, 18)
        grid = GridSpec(grid_sizes[0], grid_sizes[1])

        num_tf_plots = 3
        n_features = self.features.shape[-1]
        num_feature_plots = 3

        if n_features == 2:
            num_feature_plots = 2
        elif n_features == 1:
            num_feature_plots = 1

        spacing = int(grid_sizes[0] / num_feature_plots)
        left = int(grid_sizes[0] / 3)

        for i in range(num_feature_plots):
            start = i*spacing
            end = start + spacing
            if i == 0:
                plt.subplot(grid[start:end, :left])
                self.plot_features_1d(threshold, bins=bins)
            if i == 1:
                plt.subplot(grid[start:end, :left])
                self.plot_features_2d(threshold)
            if i == 2:
                ax = plt.subplot(grid[start:end, :left], projection='3d')
                self.plot_features_3d(threshold, ax)
            
            
        spacing = int(grid_sizes[1] / num_tf_plots)
        plt.subplot(grid[:spacing, left:])
        plt.imshow(np.mean(self.log_spectrogram, axis=-1), origin='lower',
                   aspect='auto', cmap='magma')
        plt.xticks([])
        plt.ylabel('Frequency')
        plt.title('Mixture')        

        plt.subplot(grid[2*spacing:2*spacing + spacing, left:])
        assignments = (np.max(np.argmax(self.assignments, axis=0), axis=-1)) + 1
        silence_mask = np.mean(self.log_spectrogram, axis=-1) > threshold
        assignments *= silence_mask


        plt.imshow(assignments,
                   origin='lower', aspect='auto', cmap=cmap)
        plt.xlabel('Time (frames)')
        plt.ylabel('Frequency')
        plt.title('Source assignments')

        norm = mpl.colors.Normalize(
            vmin=np.min(assignments),
            vmax=np.max(assignments)
        )
        labels = [f'Source {i}' for i in range(1, self.num_sources + 1)]
        values = list(range(np.min(assignments)+1, np.max(assignments) + 1))
        mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        _legend = [mpl.patches.Patch(color=mapper.to_rgba(0), label='Silence')]

        for value, label in zip(values, labels):
            _legend.append(mpl.patches.Patch(
                color=mapper.to_rgba(value),
                label=label
            ))
        plt.legend(handles=_legend)

        plt.subplot(grid[1*spacing:spacing + spacing, left:])
        plt.imshow(np.max(self.confidence, axis=-1) * (assignments > 0), origin='lower',
                   aspect='auto', cmap='magma')
        plt.xticks([])
        plt.ylabel('Frequency')
        plt.title('Confidence')  

        
          



    