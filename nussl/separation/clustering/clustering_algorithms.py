from .clustering_separation_base import ClusteringSeparationBase
import numpy as np
import torch
import librosa
from ..deep_mixin import DeepMixin

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

class DeepClustering(ClusteringSeparationBase, DeepMixin):
    def __init__(
        self, 
        input_audio_signal,
        model_path, 
        use_cuda=True,
        **kwargs
    ):
        
        self.device = torch.device(
            'cuda'
            if torch.cuda.is_available() and use_cuda
            else 'cpu'
        )

        self.model, self.metadata = self.load_model(model_path)
        if input_audio_signal.sample_rate != self.metadata['sample_rate']:
            input_audio_signal.resample(self.metadata['sample_rate'])

        input_audio_signal.stft_params.window_length = self.metadata['n_fft']
        input_audio_signal.stft_params.n_fft_bins = self.metadata['n_fft']
        input_audio_signal.stft_params.hop_length = self.metadata['hop_length']
        
        super().__init__(
            input_audio_signal,
            **kwargs
        )

    def project_data(self, data):
        if self.model.layers['mel_projection'].num_mels > 0:
            data = torch.from_numpy(data).to(self.device).float().unsqueeze(0)
            data = self.model.project_data(data, clamp=False)
            data = (data > 0).squeeze(0).cpu().data.numpy().astype(bool)
        return data

    def extract_features(self):
        input_data = self._preprocess()
        with torch.no_grad():
            output = self.model(input_data)
            output = {k: output[k].cpu() for k in output}

            if 'embedding' not in output:
                raise ValueError("This model is not a deep clustering model!")

            embedding = output['embedding']
            embedding_size = embedding.shape[-1]
            embedding = embedding.permute(2, 1, 0, 3)
            embedding = embedding.reshape(-1, embedding_size).data.numpy()
        return embedding