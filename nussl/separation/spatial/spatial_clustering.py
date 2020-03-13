from ..base import ClusteringSeparationBase
import numpy as np

class SpatialClustering(ClusteringSeparationBase):
    """
    Implements clustering on IPD/ILD features between the first two channels.

    IPD/ILD features are inter-phase difference and inter-level difference
    features. Sounds coming from different directions will naturally cluster
    in IPD/ILD space. 
    """
    def extract_features(self):
        ipd, ild = self.audio_signal.ipd_ild_features()
        num_channels = self.audio_signal.num_channels
        
        features = [
            [ipd for i in range(num_channels)],
            [ild for i in range(num_channels)]
        ]
        features = np.array(features).transpose(2, 3, 1, 0)   

        return features