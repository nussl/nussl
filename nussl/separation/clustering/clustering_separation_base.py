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
from copy import deepcopy

from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import scale

from nussl.deep.train.loss import DeepClusteringLoss
import torch
from sklearn.preprocessing import OneHotEncoder

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
        clustering_type='kmeans',
        enhancement_amount=0.0,
        apply_pca=False,
        num_pca_dimensions=2,
        scale_features=False,
        ref=np.max,
        order_sources_by_size=True,
    ):
        super(ClusteringSeparationBase, self).__init__(
            input_audio_signal=input_audio_signal,
            mask_type=mask_type
        )

        self.num_sources = num_sources
        self.clustering_options = (
            {} if clustering_options is None else clustering_options 
        )
        self.alpha = alpha
        self.use_librosa_stft = use_librosa_stft
        self.ref = ref
        
        allowed_clustering_types = ['kmeans', 'gmm', 'spectral_clustering']
        if clustering_type not in allowed_clustering_types:
            raise ValueError(
                f"clustering_type = {clustering_type} not allowed!" 
                f"Use one of {allowed_clustering_types}."
            )

        self.clustering_type = clustering_type
        self.clusterer = None
        self.percentile = percentile
        self.features = None
        self.enhancement_amount = enhancement_amount
        self.order_sources_by_size = order_sources_by_size
        self.apply_pca = apply_pca
        self.num_pca_dimensions = num_pca_dimensions
        self.scale_features = scale_features

    def _compute_spectrograms(self):
        self.stft = self.audio_signal.stft(
            overwrite=True,
            remove_reflection=True,
            use_librosa=self.use_librosa_stft
        )
        self.log_spectrogram = librosa.amplitude_to_db(np.abs(self.stft), ref=self.ref)

        threshold = self.log_spectrogram
        threshold = (threshold >= np.percentile(threshold, self.percentile)).astype(float)
        self.sample_weight = self.project_data(np.abs(self.stft))
        self.threshold = self.project_data(threshold).astype(bool)

    def init_clusterer(self):
        if self.clustering_type == 'kmeans':
            clusterer = KMeansConfidence
        elif self.clustering_type == 'gmm':
            clusterer = GaussianMixtureConfidence
            self.clustering_options['threshold'] = self.threshold.flatten()
        elif self.clustering_type == 'spectral_clustering':
            self.clustering_options['weights'] = self.threshold.flatten()
            clusterer = SpectralClusteringConfidence
        
        return clusterer(
            n_components=self.num_sources,
            alpha=self.alpha,
            **self.clustering_options
        )

    def project_data(self, data):
        return data

    def extract_features(self):
        raise NotImplementedError()

    def _scale_features(self, features):
        features = scale(features, axis=0)
        return features

    def cluster_features(self, features, clusterer):
        if self.clustering_type == 'kmeans':
            sample_weight = self.sample_weight.flatten()
            sample_weight = sample_weight[self.threshold.flatten()]
            clusterer.fit(
                features[self.threshold.flatten()], sample_weight=sample_weight
            )
        elif self.clustering_type != 'spectral_clustering':
            clusterer.fit(features[self.threshold.flatten()])
        else:
            clusterer.fit(features)
        assignments, confidence = clusterer.predict_and_get_confidence(features)
        return assignments, confidence

    def postprocess(self, assignments, confidence):
        assignments = assignments.reshape(self.stft.shape + (self.num_sources,))
        confidence = confidence.reshape(self.stft.shape)
        assignments = assignments.transpose(3, 0, 1, 2)
        return assignments, confidence

    def _order_sources_by_size(self, estimates):
        # orders sources by size, largest to smallest
        source_fractions = [m.mask.sum() for m in self.masks]
        estimates = [
            x for _, x in sorted(zip(source_fractions, estimates), reverse=True)
        ]
        return estimates

    def enhance(self, estimates, amount = 0.5):
        # an odd trick to trade SDR for SIR
        source_fractions = np.array([m.mask.sum() for m in self.masks])
        source_fractions /= source_fractions.sum()
        smallest_signal = estimates[np.argmin(source_fractions)]
        source_fractions = source_fractions.tolist()
        
        # How much to weight confidence by. Higher amounts lead to a sparser enhanced
        # source, with a trade-off between leakage between estimates. 0.5 seems to be a 
        # good amount, but enhancement_amount is set to 0 for backwards compat.
        confidence_mask = self.confidence ** amount
        confidence_mask = confidence_mask / np.maximum(confidence_mask.max(), 1e-7)
        
        confidence_mask = masks.SoftMask(confidence_mask)
        # In the smallest signal, get rid of points we are not confident in.
        # Idea being that the prior should take precedence for points that we are not
        # confident in their time-frequency label.
        enhanced = smallest_signal.apply_mask(confidence_mask)
        enhanced.istft(overwrite=True, truncate_to_length=self.audio_signal.signal_length)
        
        residual = (smallest_signal - enhanced) / (len(estimates) - 1)
        new_signals = []
        
        for frac, estimate in zip(source_fractions, estimates):
            if frac == np.max(source_fractions):
                # Add residual to the biggest source
                new_signals.append(estimate + residual)
            elif frac == np.min(source_fractions):
                # Use enhanced source in place of original source
                new_signals.append(enhanced)
            else:
                # Keep other sources the same
                new_signals.append(estimate)
        return new_signals

    def run(self, features=None):
        """

        Returns:

        """
        self._compute_spectrograms()
        if self.clusterer is None:
            self.clusterer = self.init_clusterer()
        if features is not None:
            self.features = features
        else:
            self.features = self.extract_features()
            
            if self.scale_features:
                self.features = self._scale_features(self.features)

            if self.apply_pca:
                self.features, _ = self.project_embeddings(
                    self.num_pca_dimensions
                )
            
        self.assignments, self.confidence = self.cluster_features(
            self.features, self.clusterer
        )
        self.assignments, self.confidence = self.postprocess(
            self.assignments, self.confidence
        )

        if self.mask_type == self.BINARY_MASK:
            self.assignments = (self.assignments == self.assignments.max(axis=0, keepdims=True)).astype(float)
        
        self.masks = []
        
        for i in range(self.num_sources):
            mask = self.assignments[i, :, :, :]
            mask = masks.SoftMask(mask)
            self.masks.append(mask)
        self.result_masks = self.masks
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
        self.sources = self.enhance(self.sources, amount=self.enhancement_amount)
        if self.order_sources_by_size:
            self.sources = self._order_sources_by_size(self.sources)
        return self.sources

    def get_lda_confidence(self):
        features = self.features
        assignments = self.clusterer.predict(features)

        ohe = OneHotEncoder(categories='auto', sparse=False)
        assignments = ohe.fit_transform(assignments.reshape(-1, 1))
        assignments = torch.from_numpy(assignments).unsqueeze(0).float()

        features = torch.from_numpy(features).unsqueeze(0).float()
        weights = self._preprocess()['magnitude_spectrogram']
        loss = DeepClusteringLoss()
        return 1 - loss(features, assignments, weights).item()

    def get_overall_confidence(self, threshold=99, n_samples=1000, verbose=False):
        if self.confidence is None or self.log_spectrogram is None:
            raise RuntimeError('Must do separator.run() first before calling this!')
        
        weights = np.percentile(self.log_spectrogram, threshold)
        weights = self.log_spectrogram >= weights
        posterior_confidence = np.average(
            self.confidence, weights=weights
        )

        _features = self.features[self.threshold.flatten()]
        n_samples = min(n_samples, _features.shape[0])
        sampled = np.random.choice(_features.shape[0], n_samples, replace=False)

        _features = _features[sampled, :]
        labels = self.clusterer.predict(_features)
        
        source_shares = [(labels == i).sum() for i in range(self.num_sources)]
        source_shares /= sum(source_shares)
        source_loudness_confidence = source_shares.min()

        if len(np.unique(labels)) > 1:
            silhoettes = (silhouette_samples(_features, labels) + 1) / 2
            silhoette_confidence = silhoettes.mean()
        else:
            silhoette_confidence = posterior_confidence

        
        if verbose:
            print(posterior_confidence, silhoette_confidence, source_loudness_confidence)

        lda_confidence = self.get_lda_confidence()

        overall_confidence = {
            'posterior_confidence': posterior_confidence,
            'silhoette_confidence': silhoette_confidence,
            'source_loudness_confidence': source_loudness_confidence,
            'lda_confidence': lda_confidence
        }

        return overall_confidence

    def set_audio_signal(self, new_audio_signal):
        input_audio_signal = deepcopy(new_audio_signal)
        self.audio_signal = input_audio_signal
        self.original_length = input_audio_signal.signal_length
        self.original_sample_rate = input_audio_signal.sample_rate
        self.clusterer = None
        return input_audio_signal

    def project_embeddings(self, num_dimensions, threshold=-80):
        """
        Does a PCA projection of the embedding space
        Args:
            num_dimensions (int): Number of dimensions to project down to
            threshold (float): Cutoff for plotting points

        Returns:

        """
        transform = PCA(n_components=num_dimensions)
        mask = (self.log_spectrogram >= threshold).astype(float)      
        mask = self.project_data(mask)
        mask = mask.reshape(-1) > 0
        _embedding = self.features[mask]
        output_transform = transform.fit_transform(_embedding)
        return output_transform, transform

    def plot_features_1d(self, threshold, plot_means=True, bins=150):
        import matplotlib.pyplot as plt
        output_transform, transform = self.project_embeddings(1, threshold=threshold)
        plt.hist(output_transform, bins=bins)
        plt.xlabel('PCA dim 1')
        plt.ylabel('Count')
        plt.title('Embedding visualization (1D)')

        if hasattr(self.clusterer, 'cluster_centers_') and plot_means:
            means = transform.transform(self.clusterer.cluster_centers_)
            for i in range(means.shape[0]):
                plt.axvline(means[i], color='r')



    def plot_features_2d(self, threshold):
        import matplotlib.pyplot as plt
        output_transform, transform = self.project_embeddings(2, threshold=threshold)
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

    def plot_features_3d(self, threshold, ax, max_points=5000):
        import pandas as pd
        output_transform, transform = self.project_embeddings(
            3,
            threshold=threshold,
        )

        rows_to_sample = np.random.choice(
            output_transform.shape[0], max_points, replace=False)
        output_transform = output_transform[rows_to_sample]

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

    def plot(self, cmap='Blues', bins=150, max_points=5000):
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
            max(70, self.percentile)
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
                self.plot_features_3d(threshold, ax, max_points=max_points)
            
            
        spacing = int(grid_sizes[1] / num_tf_plots)
        plt.subplot(grid[:spacing, left:])
        plt.imshow(np.mean(self.log_spectrogram, axis=-1), origin='lower',
                   aspect='auto', cmap='magma')
        plt.xticks([])
        plt.ylabel('Frequency')
        plt.title('Mixture')        

        plt.subplot(grid[2*spacing:2*spacing + spacing, left:])
        assignments = (np.max(np.argmax(self.assignments, axis=0), axis=-1)) + 1
        silence_mask = (
            np.mean(self.log_spectrogram, axis=-1) > 
            np.percentile(self.log_spectrogram, min(self.percentile, 50))
        )
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

        confidence = np.max(self.confidence, axis=-1)
        if confidence.shape == silence_mask.shape:
            confidence *= silence_mask

        plt.subplot(grid[1*spacing:spacing + spacing, left:])
        plt.imshow(confidence, origin='lower',
                   aspect='auto', cmap='magma', vmin=0.0, vmax=1.0)
        plt.xticks([])
        plt.ylabel('Frequency')
        plt.title('Confidence')