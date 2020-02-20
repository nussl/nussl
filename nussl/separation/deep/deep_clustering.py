from .clustering_separation_base import ClusteringSeparationBase
import numpy as np
import torch
from ..deep_mixin import DeepMixin
from copy import deepcopy

class DeepClustering(ClusteringSeparationBase, DeepMixin):
    def __init__(
        self, 
        input_audio_signal,
        model_path,
        metadata=None,
        extra_modules=None,
        use_cuda=False,
        **kwargs
    ):
        self.device = torch.device(
            'cuda'
            if torch.cuda.is_available() and use_cuda
            else 'cpu'
        )
        
        self.metadata = metadata
        self.model, self.metadata = self.load_model(model_path, extra_modules=extra_modules)

        input_audio_signal = self.set_audio_signal(input_audio_signal)

        sample_rate = self.metadata['sample_rate']
        num_mels = self.model.layers['mel_projection'].num_mels
        num_frequencies = (self.metadata['n_fft'] // 2) + 1
        filter_bank = None

        if num_mels > 0:
            weights = self.model.layers['mel_projection'].transform.weight.data.cpu().numpy()
            filter_bank = np.linalg.pinv(weights.T)

        self.filter_bank = filter_bank
        super().__init__(input_audio_signal, **kwargs)
        
    def set_audio_signal(self, new_audio_signal):
        input_audio_signal = deepcopy(new_audio_signal)
        if input_audio_signal.sample_rate != self.metadata['sample_rate']:
            input_audio_signal.resample(self.metadata['sample_rate'])
        input_audio_signal.stft_params.window_length = self.metadata['n_fft']
        input_audio_signal.stft_params.n_fft_bins = self.metadata['n_fft']
        input_audio_signal.stft_params.hop_length = self.metadata['hop_length']
        input_audio_signal = super().set_audio_signal(input_audio_signal)
        return input_audio_signal

    def postprocess(self, assignments, confidence):
        if self.filter_bank is not None:
            shape = (self.filter_bank.shape[0], -1, self.stft.shape[-1])
            assignments = assignments.reshape(shape + (self.num_sources,))
            confidence = confidence.reshape(shape)

            assignments = assignments.transpose()
            confidence = confidence.transpose()
                        
            assignments = np.dot(assignments, self.filter_bank) + 1e-6
            assignments = np.clip(assignments, 0.0, 1.0) 
            assignments /= np.sum(assignments, axis=0)

            confidence = np.dot(confidence, self.filter_bank)
            confidence += np.abs(confidence.min())
            
            assignments = assignments.transpose()
            confidence = confidence.transpose()
            assignments = assignments.transpose(3, 0, 1, 2)
        else:
            assignments, confidence = super().postprocess(assignments, confidence)

        return assignments, confidence

    def project_data(self, data):
        if self.model.layers['mel_projection'].num_mels > 0:
            data = self._format(data, 'rnn')
            data = torch.from_numpy(data).to(self.device).float()
            data = self.model.project_data(data, clamp=False)
            data = data.squeeze(-1).permute(2, 1, 0)
            data = (data > 0).cpu().data.numpy().astype(bool)
        return data

    def extract_features(self):
        input_data = self._preprocess()
        with torch.no_grad():
            output = self.model(input_data)
            if 'embedding' not in output:
                raise ValueError("This model is not a deep clustering model!")
            embedding = output['embedding']
            embedding_size = embedding.shape[-1]
            embedding = embedding.squeeze(-2)
            embedding = embedding.permute(2, 1, 0, 3)
            embedding = embedding.reshape(-1, embedding_size)
            embedding = embedding.data.cpu().numpy()
        return embedding

    def make_audio_signals(self):
        signals = super().make_audio_signals()
        residual = (self.audio_signal - sum(signals)).audio_data * (1 / len(signals))
        for signal in signals:
            signal.audio_data += residual
            if signal.sample_rate != self.original_sample_rate:
                signal.resample(self.original_sample_rate)
            signal.truncate_samples(self.original_length)
        return signals