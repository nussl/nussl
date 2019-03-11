from .clustering_separation_base import ClusteringSeparationBase
import numpy as np
import torch
import librosa
from ..deep_mixin import DeepMixin
from .. import FT2D, Melodia, HPSS, Repet, RepetSim
from sklearn.decomposition import TruncatedSVD

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

class PrimitiveClustering(ClusteringSeparationBase):
    def __init__(
        self, 
        input_audio_signal,
        num_cluster_sources=2,
        num_cascades=1,
        algorithms=[Melodia, FT2D, HPSS],
        reduce_noise=True,
        **kwargs
    ):
        super().__init__(
            input_audio_signal,
            **kwargs
        )
        self.algorithms = algorithms
        self.num_cascades = num_cascades
        self.num_cluster_sources = num_cluster_sources
        self.reduce_noise = reduce_noise

    def noise_reduction(self, features):
        tr = TruncatedSVD(n_components=self.num_sources)

        threshold = self.project_data(self.threshold)
        threshold = threshold.astype(bool)

        _features = features[threshold.flatten()]
        tr.fit(_features)
        output = tr.fit_transform(features)
        return output

    def extract_features_from_signal(self, signal):
        features =  []
        separations = []
        for i, algorithm in enumerate(self.algorithms):
            if hasattr(algorithm, 'cluster_features'):
                if self.num_cluster_sources > 0:
                    separator = algorithm(
                        self.audio_signal, 
                        num_sources = self.num_cluster_sources,
                        clustering_type=self.clustering_type,
                        clustering_options=self.clustering_options
                    )
            else:
                separator = algorithm(self.audio_signal)
            masks = separator.run()
            separations += separator.make_audio_signals()
            _features = []
            for s, mask in zip(separations, masks):
                _features.append(mask.mask)        
            _features = np.array(_features).transpose(1, 2, 3, 0)
            features.append(_features)
        return features, separations

    def extract_features(self):
        features = []
        current_signals = [self.audio_signal]
        for i in range(self.num_cascades):
            print(i, len(current_signals))
            separations = []
            for signal in current_signals:
                _features, _separations = self.extract_features_from_signal(signal)
                features += [float(1 / (i+1)) * f for f in  _features]
                separations += _separations
            current_signals = separations

        features = np.concatenate(features, axis=-1)
        features = features.reshape(-1, features.shape[-1])
        if self.reduce_noise:
            features = self.noise_reduction(features)
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