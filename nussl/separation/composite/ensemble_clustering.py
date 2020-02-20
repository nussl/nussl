from .clustering_separation_base import ClusteringSeparationBase
import numpy as np
from .. import FT2D, Melodia, HPSS, Repet, RepetSim, MultichannelWienerFilter
from sklearn.preprocessing import scale
from copy import deepcopy

class EnsembleClustering(ClusteringSeparationBase):
    def __init__(
        self, 
        input_audio_signal,
        algorithms,
        num_cascades=1,
        scale_features=False,
        **kwargs
    ):
        super().__init__(
            input_audio_signal,
            **kwargs
        )
        self.original_stft_params = deepcopy(self.audio_signal.stft_params)
        self.algorithms = [a[0] for a in algorithms]
        self.algorithm_parameters = [a[1] if len(a) > 1 else {} for a in algorithms]
        self.algorithm_returns = [a[2] if len(a) > 2 else [] for a in algorithms]
        self.num_cascades = num_cascades
        self.scale_features = scale_features
        self.separators = self.setup_algorithms()

    def setup_algorithms(self):
        separators = []
        mixture = deepcopy(self.audio_signal)
        for i, algorithm in enumerate(self.algorithms):
            stft_params = self.algorithm_parameters[i].pop('stft_params', None)
            if stft_params is not None:
                mixture.stft_data = None
                mixture.stft_params = stft_params

            separator = algorithm(
                mixture, 
                use_librosa_stft=self.use_librosa_stft, 
                **self.algorithm_parameters[i]
            )

            mixture.stft_params = self.original_stft_params
            separators.append(separator)
        return separators

    def set_audio_signal(self, new_audio_signal):
        super().set_audio_signal(new_audio_signal)
        self.setup_algorithms()

    def run_algorithm_on_signal(self, mixture, level):
        separations = []
        for i, separator in enumerate(self.separators):
            separator.run()
            signals = separator.make_audio_signals()
            if self.algorithm_returns[i]:
                signals = [signals[j] for j in self.algorithm_returns[i]]
            separations += signals
        return separations, self.separators

    def extract_features_from_signals(self, signals):
        features = []
        self.audio_signal.stft_data = None
        self.audio_signal.stft_params = self.original_stft_params
        mix_stft = np.abs(self.audio_signal.stft())
        for s in signals:
            s.stft_data = None
            s.stft_params = self.original_stft_params
            _stft = np.abs(s.stft())
            _feature = _stft / np.maximum(_stft, mix_stft + 1e-7)
            features.append(_feature)
        features = np.array(features).transpose(1, 2, 3, 0)
        return features

    def extract_features_from_separators(self, separators):
        features = []
        for i, s in enumerate(separators):
            masks = [m.mask for m in s.result_masks]
            if self.algorithm_returns[i]:
                masks = [masks[j] for j in self.algorithm_returns[i]]
            features += masks
        features = np.array(features).transpose(1, 2, 3, 0)
        return features

    def extract_features(self):
        features = []
        current_signals = [self.audio_signal]
        separators = []
        for i in range(self.num_cascades):
            separations = []
            for signal in current_signals:
                _separations, _separator = self.run_algorithm_on_signal(signal, i)
                separations += _separations
                separators += _separator
            current_signals = separations
        self.separations = separations
        features = self.extract_features_from_separators(separators)
        self._compute_spectrograms()
        features = features.reshape(-1, features.shape[-1])
        if self.scale_features:
            features = scale(features, axis=0)
        return features