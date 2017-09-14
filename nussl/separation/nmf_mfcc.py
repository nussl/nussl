#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sklearn.cluster
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
        Non Negative Matrix Factorization using K-Means Clustering on MFCC (NMF MFCC) is a source separation
        algorithm that runs Transformer NMF on the magnitude spectrogram of an input audio signal.
        It uses K means clustering to cluster the templates and activations returned by the NMF. The dot product of the
        clustered templates and activations results in a magnitude spectrogram only containing a separated source.
        This is used to create a Binary Mask object, and the whole process can be applied for each cluster to
        return a list of Audio Signal objects corresponding to each separated source.

        References:
            * `Mel Frequency Cepstral Coefficients (MFCC) <https://en.wikipedia.org/wiki/Mel-frequency_cepstrum>`_

            * `scikit-learn KMeans <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_

            * Spiertz, Martin, and Volker Gnann. "Source-filter based clustering for monaural blind source separation."
               Proceedings of the 12th International Conference on Digital Audio Effects. 2009.

        See Also:
            :ref:`The NMF MFCC Demo Example <nmf_mfcc_demo>`

        Parameters:
            input_audio_signal (:class:`audio_signal.AudioSignal`): The :class:`audio_signal.AudioSignal` object that
             NMF MFCC will be run on. This makes a copy of ``input_audio_signal``
            num_sources (int): Number of sources to find.
            num_templates (int): Number of template vectors to used in NMF. Defaults to 50
            distance_measure (str): The type of distance measure to use in NMF - euclidean or divergence.
             Defaults to euclidean.
            num_iterations (int): The number of iterations to go through in NMF. Defaults to 50.
            random_seed (int): The seed to use in the numpy random generator. Default uses no seed.
            kmeans_kwargs (dict): The kwargs for KMeans parameters. Default is none.
            convert_to_mono (bool): Given a stereo signal, convert to mono. Default set to False.
            mask_type (object): A soft or binary mask object used for the source separation.
             Defaults to Binary.
            mfcc_range (int/list/tuple): The range of mfcc to use for clustering. Defaults to 1:14.
            n_mfcc (int): The max number of mfccs to use. Defaults to 20.

        Attributes:
            input_audio_signal (:class:`audio_signal.AudioSignal`): The :class:`audio_signal.AudioSignal` object that
              NMF MFCC will be run on. This makes a copy of ``input_audio_signal``
            clusterer (:obj:`object`): A KMeans object for clustering the templates and activations.
            signal_stft (:obj:`np.matrix`): The stft data for the input audio signal.
            labeled_templates (:obj:`np.array`): A Numpy array containing the labeled templates columns
                                               from the templates matrix for a particular source.
            sources (:obj:`np.array`): A Numpy array containing the list of Audio Signal objects for each source.
            masks (:obj:`np.array`): A Numpy array containing the lists of Binary Mask objects for each channel.

         Example:

        .. code-block:: python
            :linenos:
            # Initialize input signal
            signal = nussl.AudioSignal(path_to_input_file='input_name.wav')

            # Seeding NMF and KMeans with 0 (no kmeans_kwarg initialized, so random_seed is used)
            nmf_mfcc =  nussl.NMF_MFCC(signal, num_sources=2, random_seed=0)

            # Use individual seeding
            nmf_mfcc =  nussl.NMF_MFCC(signal, num_sources=2, random_seed=0, kmeans_kwargs = {random_state: 0})

             # Only set random_seed and not random_state
            nmf_mfcc =  nussl.NMF_MFCC(signal, num_sources=2, random_seed=0, kmeans_kwargs = {random_state: None})


        """
    def __init__(self, input_audio_signal, num_sources, num_templates=50, distance_measure='euclidean',
                 num_iterations=50, random_seed=None, kmeans_kwargs=None, convert_to_mono=False,
                 mask_type=mask_separation_base.MaskSeparationBase.BINARY_MASK, mfcc_range=(1, 14), n_mfcc=20):
        super(NMF_MFCC, self).__init__(input_audio_signal=input_audio_signal, mask_type=mask_type)

        self.num_sources = num_sources
        self.num_templates = num_templates
        self.distance_measure = distance_measure
        self.num_iterations = num_iterations
        self.random_seed = random_seed
        self.kmeans_kwargs = kmeans_kwargs
        self.convert_to_mono = convert_to_mono
        self.n_mfcc = n_mfcc
        self.mask_type = mask_type

        self.signal_stft = None
        self.input_audio_signal = input_audio_signal
        self.labeled_templates = None
        self.sources = []
        self.masks = []

        # Convert the stereo signal to mono if indicated
        if self.convert_to_mono:
            self.input_audio_signal.to_mono(overwrite=True, remove_channels=False)

        # Set the MFCC range
        if isinstance(mfcc_range, int) and mfcc_range < n_mfcc:
            self.mfcc_start, self.mfcc_end = 1, mfcc_range
        elif isinstance(mfcc_range, (tuple, list)) and len(mfcc_range) == 2:
            self.mfcc_start, self.mfcc_end = mfcc_range[0], mfcc_range[1]
        else:
            raise ValueError('mfcc_range is not set correctly! Must be a tuple or list with min and max, or int (max)')

        # If kmeans_kwargs does not include the 'random_state', use the random_seed instead. Else, use the value
        # provided in the dictionary. If kmeans_kwargs is None, use the random_seed.
        self.kmeans_random_seed = kmeans_kwargs.pop('random_state', random_seed) \
            if isinstance(kmeans_kwargs, dict) else random_seed

        # Initialize the K Means clusterer
        if isinstance(self.kmeans_kwargs, dict):
            self.clusterer = sklearn.cluster.KMeans(n_clusters=self.num_sources, random_state=self.kmeans_random_seed,
                                                    **kmeans_kwargs)
        else:
            self.clusterer = sklearn.cluster.KMeans(n_clusters=self.num_sources, random_state=self.kmeans_random_seed)

    def run(self):
        """ This function calls TransformerNMF on the magnitude spectrogram of each channel in the input audio signal.
        The templates and activation matrices returned are clustered using K-Means clustering. These clusters are used
        to create mask objects for each source. Note: The masks are not returned in a particular order, but they are in
        the same order for each channel.

        Returns:
            masks (:obj:`list`): A list of :obj:`MaskBase`-derived objects for each source.
            (to get a list of :obj:`AudioSignal`-derived objects run :func:`make_audio_signals`)

        Example:

        .. code-block:: python
            :linenos:

            signal = nussl.AudioSignal(path_to_input_file='input_name.wav')

            # Set up and run NMF MFCC
            nmf_mfcc =  nussl.NMF_MFCC(signal, num_sources=2) # Returns a binary mask by default
            masks = nmf_mfcc.run()

            # Get audio signals
            sources = nmf_mfcc.make_audio_signals()

            # Output the sources
            for i, source in enumerate(sources):
                output_file_name = str(i) + '.wav'
                source.write_audio_to_file(output_file_name)
        """
        self.masks = []
        self.input_audio_signal.stft_params = self.stft_params
        self.signal_stft = self.input_audio_signal.stft()

        for i in range(self.input_audio_signal.num_channels):
            channel_stft = self.input_audio_signal.get_magnitude_spectrogram_channel(i)

            # Set up NMF and run
            nmf = transformer_nmf.TransformerNMF(input_matrix=channel_stft, num_components=self.num_templates,
                                                 seed=self.random_seed)

            # Pass these to constructor (above)
            nmf.should_use_epsilon = False
            nmf.max_num_iterations = self.num_iterations
            nmf.distance_measure = self.distance_measure
            channel_activation_matrix, channel_templates_matrix = nmf.transform()

            # Cluster the templates matrix into Mel frequencies and retrieve labels
            cluster_templates = librosa.feature.mfcc(S=channel_templates_matrix,
                                                     n_mfcc=self.n_mfcc)[self.mfcc_start:self.mfcc_end]
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
            if self.mask_type == self.BINARY_MASK:
                mask = np.round(mask)
                mask_object = masks.BinaryMask(np.array(mask))
            else:
                mask_object = masks.SoftMask(np.array(mask))
            channel_mask_list.append(mask_object)
        return channel_mask_list

    def make_audio_signals(self):
        """ Applies each mask in self.masks and returns a list of audio_signal objects for each source.

        Returns:
            self.sources (np.array): An array of audio_signal objects containing each separated source
        """
        self.sources = []
        for i in range(self.input_audio_signal.num_channels):
            channel_mask = self.masks[i]
            for j in range(self.num_sources):
                channel_stft = self.input_audio_signal.get_stft_channel(i)
                source = self.audio_signal.make_copy_with_stft_data(channel_stft, verbose=False)
                source = source.apply_mask(channel_mask[j])
                source.stft_params = self.stft_params
                source.istft(overwrite=True, truncate_to_length=self.audio_signal.signal_length)
                self.sources.append(source)
        return self.sources
