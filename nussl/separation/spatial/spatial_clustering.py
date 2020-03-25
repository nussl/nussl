import numpy as np

from ..base import ClusteringSeparationBase


class SpatialClustering(ClusteringSeparationBase):
    """
    Implements clustering on IPD/ILD features between the first two channels.

    IPD/ILD features are inter-phase difference and inter-level difference
    features. Sounds coming from different directions will naturally cluster
    in IPD/ILD space. 

    Subclasses ClusteringSeparationBase which actually handles all of the
    clustering functionality behind this function.
    """
    def extract_features(self):
        ipd, ild = self.audio_signal.ipd_ild_features()
        num_channels = self.audio_signal.num_channels
        
        features = [
            [ipd for _ in range(num_channels)],
            [ild for _ in range(num_channels)]
        ]
        features = np.array(features).transpose(2, 3, 1, 0)   

        return features
