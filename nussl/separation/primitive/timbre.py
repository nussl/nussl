import numpy as np
import librosa

from ..base import ClusteringSeparationBase, NMFMixin


class TimbreClustering(ClusteringSeparationBase, NMFMixin):
    """
    Implements separation by timbre via NMF with MFCC clustering. The
    steps are:

    1. Factorize the magnitude spectrogram of the mixture with NMF.
    2. Take MFCC coefficients of each component.
    3. Express each time-frequency bin as a combination of components.
    4. The features for each time-frequency bin are the weighted combination
       of the MFCCs of each component.
    5. Cluster each time-frequency bin based on these features.
    
    Args:
        input_audio_signal (AudioSignal): Signal to separate.
        n_components (int): Number of components to use in the NMF
          model. Corresponds to number of spectral templates.
        n_mfcc (int): Number of MFCC coefficients to use. Defaults to 13.
        nmf_args (dict): Dictionary containing keyword arguments for `NMFMixin.fit`.
        kwargs (dict): Extra keyword arguments are passed to ClusteringSeparationBase.
    """

    def __init__(self, input_audio_signal, num_sources, n_components, n_mfcc=13,
                 nmf_kwargs=None, **kwargs):
        self.n_components = n_components
        self.nmf_kwargs = {} if nmf_kwargs is None else nmf_kwargs
        self.n_mfcc = n_mfcc

        super().__init__(
            input_audio_signal=input_audio_signal,
            num_sources=num_sources, **kwargs)

    def extract_features(self):
        model, components, activations = self.fit(
            self.audio_signal, self.n_components, **self.nmf_kwargs)
        mel = librosa.feature.melspectrogram(
            S=(components.T ** 2), sr=self.sample_rate)
        mfcc = librosa.feature.mfcc(S=mel, n_mfcc=self.n_mfcc).T

        activations = activations.reshape(self.n_components, -1, 1)
        components = components.reshape(self.n_components, 1, -1)

        expansion = activations @ components
        expansion = expansion.transpose()
        features = expansion @ mfcc

        features = features.reshape(self.stft.shape + (-1,))
        norm = np.linalg.norm(features, axis=-1, keepdims=True)
        features = features / (norm + 1e-8)

        return features
