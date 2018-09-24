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

from ..transformers import TransformerDeepClustering
from sklearn.decomposition import PCA
import mask_separation_base
import masks


class DeepClustering(mask_separation_base.MaskSeparationBase):
    """Implements deep clustering for source separation, using PyTorch.

    Deep clustering is a deep learning approach to source separation. It takes as input a
    mel-spectrogram representation of an audio mixture. Each time-frequency bin is mapped into
    an K-dimensional embedding. The model works out so that time-frequency bins that are dominated
    by different sources map to embeddings that are distant, and bins that are dominated by the
    same source map to embeddings that are near. The sources are then recovered using K-Means
    clustering on the embedding space.

    References:

    Hershey, J. R., Chen, Z., Le Roux, J., & Watanabe, S. (2016, March).
    Deep clustering: Discriminative embeddings for segmentation and separation.
    In Acoustics, Speech and Signal Processing (ICASSP),
    2016 IEEE International Conference on (pp. 31-35). IEEE.

    Luo, Y., Chen, Z., Hershey, J. R., Roux, J. L., & Mesgarani, N. (2016).
    Deep Clustering and Conventional Networks for Music Separation: Stronger Together.
    arXiv preprint arXiv:1611.06265.

    Example:
        music = AudioSignal("path/to/input.wav", offset=45, duration=20)

        music.stft_params.window_length = 2048
        music.stft_params.hop_length = 512

        separation = DeepClustering(music, num_sources = 2)
        masks = separation.run()
        sources = separation.make_audio_signals()
        plt.figure(figsize=(20, 8))
        separation.plot()
        plt.tight_layout()
        plt.show()
    """
    def __init__(self, input_audio_signal,
                 mask_type=mask_separation_base.MaskSeparationBase.SOFT_MASK,
                 model_path='/media/ext/models/deep_clustering_vocal_44k_long.model',
                 num_sources=2,
                 num_layers=4,
                 hidden_size=500,
                 max_distance=1,
                 embedding_size=20,
                 num_mels=150,
                 do_mono=False,
                 resample_rate=44100,
                 use_librosa_stft=False,
                 cutoff=-40):

        if not torch_okay:
            raise ImportError('Cannot import pytorch! Install pytorch to continue.')

        super(DeepClustering, self).__init__(input_audio_signal=input_audio_signal,
                                             mask_type=mask_type)

        self.resample_rate = resample_rate
        if self.audio_signal.sample_rate != self.resample_rate:
            self.audio_signal.resample(self.resample_rate)

        self.use_librosa_stft = use_librosa_stft
        self.num_mels = num_mels
        self.num_sources = num_sources
        self.num_fft = self.audio_signal.stft_params.n_fft_bins
        self.mel_filter_bank = librosa.filters.mel(self.resample_rate,
                                                   self.num_fft, self.num_mels).T
        self.inverse_mel_filter_bank = np.linalg.pinv(self.mel_filter_bank)
        self.max_distance = max_distance
        self.use_cuda = torch.cuda.is_available()
        if not self.use_cuda:
            warnings.warn('CUDA not available, running on CPU may be slow')

        self.stft = None
        self.mel_spectrogram = None
        self.silence_mask = None
        self.sources = None
        self.assignments = None
        self.centroids = None
        self.masks = []
        self.cutoff = cutoff
        self.model = TransformerDeepClustering(num_layers=num_layers,
                                               hidden_size=hidden_size,
                                               embedding_size=embedding_size)
        if self.use_cuda:
            self.model = self.model.cuda()

        self.load_model(model_path)
        self.clusterer = KMeans(n_clusters=self.num_sources)
        self.embeddings = None

        self.do_mono = do_mono

        if self.do_mono:
            self.audio_signal.to_mono(overwrite=True)

    def load_model(self, model_path):
        """
        Loads the model at specified path ``model_path``
        Args:
            model_path:

        Returns:

        """
        self.model.load_state_dict(torch.load(model_path,
                                              map_location=lambda storage, loc: storage))
        self.model.eval()

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

    def deep_clustering(self):
        """

        Returns:

        """
        input_data = Variable(torch.FloatTensor(self.mel_spectrogram))

        if self.use_cuda:
            input_data = input_data.cuda()

        if self.embeddings is None:
            embeddings = self.model(input_data)
            self.embeddings = embeddings.view(-1, embeddings.size(-1)).cpu().data.numpy()
        self.clusterer.fit(self.embeddings)

        assignments = self.clusterer.labels_ + 1
        self.assignments = assignments.reshape(self.mel_spectrogram.shape)
        self.centroids = self.clusterer.cluster_centers_

    def _extract_masks(self, ch):
        if self.audio_signal.stft_data is None:
            raise ValueError('Cannot extract masks with no signal_stft data')

        channel_mask_list = []

        for cluster_index in range(1, self.num_sources + 1):
            if self.mask_type == self.BINARY_MASK:
                mask = ((self.silence_mask[ch, :, :] * self.assignments[ch, :, :]) == cluster_index)

            elif self.mask_type == self.SOFT_MASK:
                d = (self.embeddings - self.centroids[cluster_index-1])**2
                distances = np.sqrt(np.sum(d, axis=-1))
                distances += distances.min()
                distances /= (distances.max() + 1e-7)
                distances[distances > self.max_distance] = 1
                distances = 1 - distances.reshape(self.mel_spectrogram.shape)
                mask = (self.silence_mask[ch, :, :] * distances[ch, :, :])
            else:
                raise ValueError('Unknown mask type {}!'.format(self.mask_type))

            mask = np.dot(mask, self.inverse_mel_filter_bank).T
            mask += np.abs(mask.min())
            mask /= (np.max(mask) + 1e-7)
            channel_mask_list.append(mask)

        if self.mask_type == self.SOFT_MASK:
            data = np.stack(channel_mask_list, axis=-1)
            original_shape = data.shape
            data = data.reshape(-1, 2)**3
            data = (data.T / np.sum(data, axis=-1)).T
            data = data.reshape(original_shape)
            channel_mask_list = [data[:, :, i] for i in range(self.num_sources)]

        return channel_mask_list

    def generate_mask(self, ch, assignments):
        """
            Takes binary Mel spectrogram assignments and generates mask
        """
        if self.audio_signal.stft_data is None:
            raise ValueError('Cannot extract masks with no signal_stft data')

        mask = (self.silence_mask[ch, :, :] * assignments)
        mask = np.dot(mask, self.inverse_mel_filter_bank).T
        mask += np.abs(mask.min())
        mask /= (np.max(mask) + 1e-7)
        mask = np.round(mask)

        # mask = np.dstack([mask, mask])

        return masks.BinaryMask(mask)

    def run(self):
        """

        Returns:

        """
        self._compute_spectrograms()
        self.deep_clustering()

        uncollated_masks = []
        for i in range(self.audio_signal.num_channels):
            uncollated_masks += self._extract_masks(i)

        collated_masks = [np.dstack([uncollated_masks[s + ch * self.num_sources]
                                     for ch in range(self.audio_signal.num_channels)])
                          for s in range(self.num_sources)]

        self.masks = []

        for mask in collated_masks:
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

    def project_embeddings(self, num_dimensions):
        """
        Does a PCA projection of the embedding space
        Args:
            num_dimensions:

        Returns:

        """
        transform = PCA(n_components=num_dimensions)
        output_transform = transform.fit_transform(self.embeddings)
        return output_transform

    def plot(self, **kwargs):
        """ Plots relevant information for deep clustering onto the active figure,
            given by matplotlib.pyplot.figure()
            outside of this function. The three plots are:
                1. PCA of emeddings onto 2 dimensions for visualization
                2. The mixture mel-spectrogram.
                3. The source assignments of each tf-bin in the mixture spectrogram.
        Returns:
            None
        """
        grid = GridSpec(6, 10)
        output_transform = self.project_embeddings(2)
        plt.subplot(grid[:3, 3:])
        plt.imshow(np.mean(self.mel_spectrogram, axis=0).T, origin='lower',
                   aspect='auto', cmap='magma')
        plt.xticks([])
        plt.ylabel('Frequency (mel)')
        plt.title('Mixture')

        plt.subplot(grid[1:-1, :3])

        xmin = output_transform[:, 0].min()
        xmax = output_transform[:, 0].max()
        ymin = output_transform[:, 1].min()
        ymax = output_transform[:, 1].max()

        plt.hexbin(output_transform[:, 0], output_transform[:, 1], bins='log', gridsize=100)
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xlabel('PCA dim 1')
        plt.ylabel('PCA dim 2')
        plt.title('Embedding visualization')

        plt.subplot(grid[3:, 3:])
        plt.imshow(np.max(self.silence_mask * self.assignments, axis=0).T,
                   origin='lower', aspect='auto', cmap='Greys')
        plt.xlabel('Time (frames)')
        plt.ylabel('Frequency (mel)')
        plt.title('Source assignments')
