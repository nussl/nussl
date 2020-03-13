import copy
from ... import ml
from . import SeparationException, MaskSeparationBase
import numpy as np
from sklearn import preprocessing

ALLOWED_CLUSTERING_TYPES = ['KMeans', 'GaussianMixture', 'MiniBatchKMeans']

class ClusteringSeparationBase(MaskSeparationBase):
    """
    A base class for any clustering-based separation approach. Subclasses 
    of this class must implement just one function to use it: `extract_features`.
    This function should uses the internal variables of the class to 
    extract the appropriate time-frequency features of the signal. These 
    time-frequency features will then be clustered by `cluster_features`. 
    Masks will then be produced by the run function and applied to the 
    audio signal to produce separated estimates.
    
    Args:
        input_audio_signal: (`AudioSignal`) An AudioSignal object containing the 
          mixture to be separated.
        num_sources (int): Number of sources to cluster the features of and separate
          the mixture.
        clustering_type (str): One of 'KMeans', 'GaussianMixture', and 'MiniBatchKMeans'.
          The clustering approach to use on the features. Defaults to 'KMeans'.
        percentile (int): Percentile of time-frequency points to consider by loudness. 
          Audio spectrograms are very high dimensional, and louder points tend to 
          matter more than quieter points. By setting the percentile high, one can more
          efficiently cluster an auditory scene by considering only points above
          that threshold. Defaults to 90 (which means the top 10 percentile of 
          time-frequency points will be used for clustering).
        beta (float): When using KMeans, we use soft KMeans, which has an additional 
          parameter `beta`. `beta` controls how soft the assignments are. As beta 
          increases, the assignments become more binary (either 0 or 1). Defaults to 
          5.0, a value discovered through cross-validation.
        mask_type (str): Masking approach to use. Passed up to MaskSeparationBase.
        mask_threshold (float): Threshold for masking. Passed up to MaskSeparationBase.
        **kwargs (dict): Additional keyword arguments that are passed to the clustering
          object (one of KMeans, GaussianMixture, or MiniBatchKMeans).
    
    Raises:
        SeparationException: If clustering type is not one of the allowed ones, or if
          the output of `extract_features` has the wrong shape according to the STFT
          shape of the AudioSignal.
    """
    def __init__(self, input_audio_signal, num_sources, clustering_type='KMeans', 
        percentile=90, beta=5.0, mask_type='soft', mask_threshold=0.5, **kwargs):

        if clustering_type not in dir(ml.cluster):
            raise SeparationException(
                f"clustering_type = {clustering_type} not allowed!" 
                f"Use one of {ALLOWED_CLUSTERING_TYPES}."
            )

        ClusterClass = getattr(ml.cluster, clustering_type)

        self.num_sources = num_sources
        if clustering_type == 'GaussianMixture':
            self.clusterer = ClusterClass(n_components=num_sources, **kwargs)
        else:
            self.clusterer = ClusterClass(n_clusters=num_sources, **kwargs)
        self.clustering_type = clustering_type

        self.percentile = percentile
        self.features = None
        self.beta = beta

        super(ClusteringSeparationBase, self).__init__(
            input_audio_signal=input_audio_signal,
            mask_type=mask_type,
            mask_threshold=mask_threshold
        )

    def _preprocess_audio_signal(self):
        """
        Preprocess the audio signal object - takes STFTs, sets features to None 
        (new audio signal means new features), finds the time frequency points above
        the cutoff according to the percentile.
        """
        self.features = None
        self.stft = self.audio_signal.stft()

        # get a cutoff using the percentile
        self.cutoff = np.percentile(np.abs(self.stft), self.percentile)
        self.tf_point_over_cutoff = np.abs(self.stft) >= self.cutoff

    def extract_features(self):
        """
        This function should be implemented by the subclass. It should extract
        features. If the STFT shape is `(n_freq, n_time, n_chan)`, the output of this
        function should be `(n_freq, n_time, n_chan, n_features)`.
        """
        raise NotImplementedError()


    def cluster_features(self, features, clusterer):
        """
        Clusters each time-frequency point according to features for each
        time-frequency point. Features should be on the last axis.

        Features should come in in the shape:
          `(..., n_features)`
        
        Args:
            features (np.ndarray): Features to cluster, for each time-frequency point.
            clusterer (object): Clustering object to use.
        
        Returns:
            np.ndarray: Responsibilities for each cluster for each time-frequency point. 
        """
        shape = features.shape
        features_to_fit = features.reshape(-1, shape[-1])
        features_to_fit = features_to_fit[self.tf_point_over_cutoff.flatten(), :]
        clusterer.fit(features_to_fit)

        if 'KMeans' in self.clustering_type:
            distances = clusterer.transform(features.reshape(-1, shape[-1]))
            distances = np.exp(-self.beta * distances)
            responsibilities = distances / distances.sum(axis=-1, keepdims=True)
        if 'GaussianMixture' in self.clustering_type:
            responsibilities = clusterer.predict_proba(features.reshape(-1, shape[-1]))

        responsibilities = responsibilities.reshape(shape[:-1] + (-1,))
        return responsibilities

    def run(self, features=None):
        """
        Clusters the features using the chosen clustering algorithm.
        
        Args:
            features (np.ndarray, optional): If features are given, then the 
              `extract_features` step will be skipped. Defaults to None (so
              `extract_features` will be run.)
        
        Raises:
            SeparationException: If features.shape doesn't match what is expected
              in the STFT of the audio signal, an exception is raised.
        
        Returns:
            list: List of Mask objects in self.result_masks.
        """
        if features is None:
            features = self.extract_features()

        if features.shape[:-1] != self.stft.shape:
            raise SeparationException(
                f"features.shape did not match stft shape along all but feature " 
                f"dimension! Got {features.shape[:-1]}, expected {self.stft.shape}.")

        responsibilities = self.cluster_features(features, self.clusterer)
        self.features = features
        self.result_masks = []

        for i in range(responsibilities.shape[-1]):
            mask_data = responsibilities[..., i]
            if self.mask_type == self.MASKS['binary']:
                mask_data = (
                    responsibilities[..., i] == responsibilities.max(axis=-1))
            mask = self.mask_type(mask_data)
            self.result_masks.append(mask)
        
        return self.result_masks
