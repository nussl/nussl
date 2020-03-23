from .. import ClusteringSeparationBase, SeparationException
import numpy as np

class EnsembleClustering(ClusteringSeparationBase):
    def __init__(self, input_audio_signal, num_sources, separators, weights=None, 
                 returns=None, num_cascades=1, extracted_feature='masks', 
                 clustering_type='KMeans', percentile=90, 
                 beta=5.0, mask_type='soft', mask_threshold=0.5, **kwargs):
        super().__init__(
            input_audio_signal, num_sources, clustering_type=clustering_type, 
            percentile=percentile, beta=beta, mask_type=mask_type, 
            mask_threshold=mask_threshold, **kwargs
        )

        self.separators = separators
        self.num_cascades = num_cascades

        if isinstance(weights, list):
            if len(weights) != len(separators):
                raise SeparationException(
                    f"len(weights) must be the same as len(separators)!")
            self.weights = weights
        else:
            self.weights = [1 for i in range(len(self.separators))]

        if isinstance(returns, list):
            if len(returns) != len(separators):
                raise SeparationException(
                    f"len(returns) must be the same as len(separators)!")
            self.returns = returns
        else:
            self.returns = None

        if extracted_feature not in ['masks', 'estimates']:
            raise SeparationException(
                f"extracted_feature must be one of ['masks', 'estimates']. "
                f"Got {extracted_feature}.")

        self.extracted_feature = extracted_feature

    def run_separators_on_mixture(self, mixture):
        estimates = []
        masks = []
        for i, separator in enumerate(self.separators):
            weight = self.weights[i]

            _estimates = separator(audio_signal=mixture)
            _names = [str(separator) for e in _estimates]
            _masks = []

            if hasattr(separator, 'result_masks'):
                _masks = separator.result_masks

            if self.returns is not None:
                returns = self.returns[i]
                _estimates = [_estimates[j] for j in returns]
                if _masks:
                    _masks = [_masks[j] for j in returns]

            for i in range(weight):
                estimates.extend(_estimates)
                masks.extend(_masks)

        return estimates, masks

    def extract_features(self):
        features = []
        current_signal = [self.audio_signal]

        for i in range(self.num_cascades):
            new_signals = []

            for _signal in current_signal:
                estimates, masks = self.run_separators_on_mixture(_signal)
                new_signals.extend(estimates)

                if self.extracted_feature == 'masks':
                    _features = self._extract_features_from_masks(masks)
                elif self.extracted_feature == 'estimates':
                    _features = self._extract_features_from_estimates(estimates)
                
                features.append(_features)
            
            current_signal = new_signals
        
        features = np.concatenate(features, axis=-1)

        return features

    def _extract_features_from_estimates(self, estimates):
        features = []
        mix_stft = np.abs(self.stft)
        for e in estimates:
            _stft = np.abs(e.stft())
            data = _stft / np.maximum(_stft, mix_stft + 1e-7)
            features.append(data)
        features = np.stack(features, axis=-1)
        return features

    def _extract_features_from_masks(self, masks):
        features = []
        for m in masks:
            mask_data = m.mask
            features.append(mask_data)
        features = np.stack(features, axis=-1)
        return features
