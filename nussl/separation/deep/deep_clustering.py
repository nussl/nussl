from ..base import ClusteringSeparationBase, DeepMixin, SeparationException
import numpy as np
import torch
from copy import deepcopy

class DeepClustering(ClusteringSeparationBase, DeepMixin):
    def __init__(self, input_audio_signal, num_sources, model_path=None, device='cpu', 
      **kwargs):
        if model_path is not None:
            self.load_model(model_path, device=device)

        super().__init__(input_audio_signal, num_sources, **kwargs)

    def extract_features(self):
        input_data = self._get_input_data_for_model()
        with torch.no_grad():
            output = self.model(input_data)
            if 'embedding' not in output:
                raise ValueError("This model is not a deep clustering model!")
            embedding = output['embedding']
            # swap back batch and sample dims
            if self.metadata['num_channels'] == 1:
                embedding = embedding.transpose(0, -2)
            embedding = embedding.squeeze(0).transpose(0, 1)
        return embedding.cpu().data.numpy()
