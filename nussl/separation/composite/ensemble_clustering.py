import numpy as np

from .. import ClusteringSeparationBase, SeparationException


class EnsembleClustering(ClusteringSeparationBase):
    """
    Run multiple separation algorithms on a single mixture and concatenate their
    masks to input into a clustering algorithm.
    
    This algorithm allows you to combine the outputs of multiple separation 
    algorithms, fusing them into a single output via clustering. It was first
    developed in [1]. When used with primitive separation algorithms, it becomes
    the PrimitiveClustering algorithm described in [1].

    References:

    [1] Seetharaman, Prem. Bootstrapping the Learning Process for Computer Audition. 
        Diss. Northwestern University, 2019.
    
    Args:
        input_audio_signal (AudioSignal): Signal to separate.

        num_sources (int): Number of sources to separate from signal.

        separators (list): List of instantiated separation algorithms that will be
          run on the input audio signal.

        weights (list, optional): Weight to give to each algorithm in the resultant
          feature vector. For example, `[3, 1]`, will repeat the features from the first
          algorithm 3 times and the second algorithm 1 time. Defaults to None - every
          algorithm gets a weight of 1.

        returns (list, optional): Which outputs of each algorithm to keep in the resultant
          feature vector. Defaults to None.

        num_cascades (int, optional): The output of each algorithm can be cascaded into 
          one another. The outputs of the first layer of algorithms will be refed to 
          each separation algorithm to create more features. Defaults to 1.

        extracted_feature (str, optional): Which feature to extract from each algorithm.
          Must be one of `['estimates', 'masks']`. `estimates` will reconstruct a soft
          mask using the output of the algorithm (useful if the algorithm is not a masking
          based separation algorithm). `masks` will use the data in the `result_masks`
          attribute of the separation algorithm. Defaults to 'masks'.

        clustering_type (str): One of 'KMeans', 'GaussianMixture', and 'MiniBatchKMeans'.
          The clustering approach to use on the features. Defaults to 'KMeans'.

        fit_clusterer (bool, optional): Whether or not to call fit on the clusterer.
          If False, then the clusterer should already be fit for this to work. Defaults
          to True.

        percentile (int, optional): Percentile of time-frequency points to consider by loudness. 
          Audio spectrograms are very high dimensional, and louder points tend to 
          matter more than quieter points. By setting the percentile high, one can more
          efficiently cluster an auditory scene by considering only points above
          that threshold. Defaults to 90 (which means the top 10 percentile of 
          time-frequency points will be used for clustering).

        beta (float, optional): When using KMeans, we use soft KMeans, which has an additional 
          parameter `beta`. `beta` controls how soft the assignments are. As beta 
          increases, the assignments become more binary (either 0 or 1). Defaults to 
          5.0, a value discovered through cross-validation.

        mask_type (str, optional): Masking approach to use. Passed up to MaskSeparationBase.

        mask_threshold (float, optional): Threshold for masking. Passed up to MaskSeparationBase.

        **kwargs (dict, optional): Additional keyword arguments that are passed to the clustering
          object (one of KMeans, GaussianMixture, or MiniBatchKMeans).
    
    Example:
        
        .. code-block:: python

            from nussl.separation import (
                primitive, 
                factorization, 
                composite, 
                SeparationException
            )

            separators = [
                primitive.FT2D(mix),
                factorization.RPCA(mix),
                primitive.Melodia(mix, voicing_tolerance=0.2),
                primitive.HPSS(mix),
            ]

            weights = [3, 3, 1, 1]
            returns = [[1], [1], [1], [0]]

            ensemble = composite.EnsembleClustering(
                mix, 2, separators, weights=weights, returns=returns)
            estimates = ensemble()
    """
    def __init__(self, input_audio_signal, num_sources, separators, weights=None, 
                 returns=None, num_cascades=1, extracted_feature='masks', 
                 clustering_type='KMeans', fit_clusterer=True, percentile=90,
                 beta=5.0, mask_type='soft', mask_threshold=0.5, **kwargs):
        super().__init__(
            input_audio_signal, num_sources, clustering_type=clustering_type, 
            percentile=percentile, fit_clusterer=fit_clusterer, beta=beta, 
            mask_type=mask_type, mask_threshold=mask_threshold, **kwargs)

        self.separators = separators
        self.num_cascades = num_cascades

        if isinstance(weights, list):
            if len(weights) != len(separators):
                raise SeparationException(
                    f"len(weights) must be the same as len(separators)!")
            self.weights = weights
        else:
            self.weights = [1 for _ in range(len(self.separators))]

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
            _names = [str(separator) for _ in _estimates]
            _masks = []

            if hasattr(separator, 'result_masks'):
                _masks = separator.result_masks

            if self.returns is not None:
                returns = self.returns[i]
                _estimates = [_estimates[j] for j in returns]
                if _masks:
                    _masks = [_masks[j] for j in returns]

            for _ in range(weight):
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

    @staticmethod
    def _extract_features_from_masks(masks):
        features = []
        for m in masks:
            mask_data = m.mask
            features.append(mask_data)
        features = np.stack(features, axis=-1)
        return features
