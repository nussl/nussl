import copy
from ... import ml
from . import SeparationException, MaskSeparationBase
import numpy as np
from sklearn import preprocessing

ALLOWED_CLUSTERING_TYPES = ['KMeans', 'GaussianMixture', 'MiniBatchKMeans']

class ClusteringSeparationBase(MaskSeparationBase):
    """
    [summary]
    
    [extended_summary]
    
    Args:
        mask_separation_base ([type]): [description]
    
    Raises:
        ValueError: [description]
        NotImplementedError: [description]
    
    Returns:
        [type]: [description]
    """
    def __init__(self, input_audio_signal, num_sources, percentile=90, beta=5.0,
      clustering_type='KMeans', mask_type='soft', mask_threshold=0.5, **kwargs):

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
        self.features = None
        self.result_masks = []

        self.stft = self.audio_signal.stft()

        # get a cutoff using the percentile
        self.cutoff = np.percentile(np.abs(self.stft), self.percentile)
        self.tf_point_over_cutoff = np.abs(self.stft) >= self.cutoff

    def extract_features(self):
        raise NotImplementedError()


    def cluster_features(self, features, clusterer):
        """
        Clusters each time-frequency point according to features for each
        time-frequency point. Features should be on the last axis.

        Features should come in in the shape:
          `(..., n_features)`
        
        Args:
            features (np.ndarray): [description]
            clusterer ([type]): [description]
        
        Returns:
            [type]: [description]
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

        for i in range(responsibilities.shape[-1]):
            mask_data = responsibilities[..., i]
            if self.mask_type == self.MASKS['binary']:
                mask_data = (
                    responsibilities[..., i] == responsibilities.max(axis=-1))
            mask = self.mask_type(mask_data)
            self.result_masks.append(mask)
        
        return self.result_masks
