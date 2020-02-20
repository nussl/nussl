from .clustering_separation_base import ClusteringSeparationBase
import numpy as np

class SpatialClustering(ClusteringSeparationBase):
    def extract_features(self):
        ipd, ild = self.audio_signal.ipd_ild_features()
        num_channels = self.audio_signal.num_channels
        
        features = [
            [ipd for i in range(num_channels)],
            [ild for i in range(num_channels)]
        ]
        features = np.array(features).transpose(2, 3, 1, 0)   
        features = features.reshape(-1, features.shape[-1])   

        return features