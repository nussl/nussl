import numpy as np

from ... import ml
from ... import AudioSignal


class NMFMixin:
    @staticmethod
    def fit(audio_signals, n_components, beta_loss='frobenius',
            l1_ratio=0.5, **kwargs):
        """
        Fits an NMF model to the magnitude spectrograms of each
        audio signal. If `audio_signals` is a list, the magnitude
        spectrograms of each signal are concatenated into a single
        data matrix to which NMF is fit. If `audio_signals`
        is a single audio signal, then NMF is fit only to the
        magnitude spectrogram for that audio signal. If any of
        the audio signals are multichannel, the channels are 
        concatenated into a single (longer) data matrix.

        Args:
            audio_signals (list or AudioSignal): AudioSignal object(s) that 
              NMF will be fit to.
            n_components (int): Number of components to use in the NMF
              module. Corresponds to number of spectral templates.
            beta_loss (float or string): String must be in 
              {'frobenius', 'kullback-leibler', 'itakura-saito'}.
              Beta divergence to be minimized, measuring the distance between X
              and the dot product WH. Note that values different from 'frobenius'
              (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
              fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
              matrix X cannot contain zeros. Used only in 'mu' solver. Defaults to 
              'frobenius'.
            l1_ratio (float): The regularization mixing parameter, with 0 <= l1_ratio <= 1.
              For l1_ratio = 0 the penalty is an elementwise L2 penalty (aka Frobenius Norm).
              For l1_ratio = 1 it is an elementwise L1 penalty.
              For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
              Defaults to 1.0 (sparse templates and activations).
            kwargs (dict): Additional keyword arguments to initialization of the NMF
              decomposition method.

        Returns:
            model (NMF): Fitted NMF model to the audio signal(s).
            components (np.ndarray): Spectral templates (n_components, n_features)
            activations (np.ndarray): Activations (n_components, n_time, n_channels)
              The shape here is as if it was like an STFT but with components as the
              features rather than frequencies of the STFT.
        """
        if isinstance(audio_signals, AudioSignal):
            audio_signals = [audio_signals]

        data = []

        n_spectrograms = 0

        for audio_signal in audio_signals:
            _data = np.abs(audio_signal.stft())
            n_spectrograms += audio_signal.num_channels
            # flip around array so frequencies are last
            _data = _data.transpose()
            # flatten first 2 axes
            _data = _data.reshape(-1, _data.shape[-1])
            data.append(_data)

        data = np.concatenate(data, axis=0)

        model = ml.NMF(n_components=n_components, l1_ratio=l1_ratio,
                       beta_loss=beta_loss, **kwargs)
        activations = model.fit_transform(data)
        activations = activations.T.reshape(n_components, -1, n_spectrograms)
        return model, model.components_, activations

    @staticmethod
    def transform(audio_signal, model):
        """
        Use an already fit model to transform the magnitude spectrogram of an 
        audio signal into components and activations. These can be multiplied to 
        reconstruct the original matrix, or used to separate out sounds that correspond
        to components in the model.
        
        Args:
            audio_signal (AudioSignal): AudioSignal object to transform with model.
            model (NMF): NMF model to separate with. Must be fitted prior to this call.
            

        Returns:
            components (np.ndarray): Spectral templates (n_components, n_features)
            activations (np.ndarray): Activations (n_components, n_time, n_channels)
              The shape here is as if it was like an STFT but with components as the
              features rather than frequencies of the STFT.
        """
        data = np.abs(audio_signal.stft())

        shape = data.shape
        data = data.transpose()
        data = data.reshape(-1, data.shape[-1])
        activations = model.transform(data).T

        activations = activations.reshape((model.n_components,) + shape[1:])
        return model.components_, activations

    @staticmethod
    def inverse_transform(components, activations):
        """
        Reconstructs the magnitude spectrogram by matrix multiplying the components 
        with the activations. Components and activations are considered to be 2D matrices, 
        but if they are more, then the first dimension is interpreted to be the batch 
        dimension.
        
        Args:
            components (np.ndarray): Spectral templates (n_components, n_features)
            activations (np.ndarray): Activations (n_components, n_time, n_channels)
              The shape here is as if it was like an STFT but with components as the
              features rather than frequencies of the STFT.
        """
        activations = activations.transpose()
        shape = activations.shape

        reconstruction = activations @ components
        reconstruction = reconstruction.reshape(shape[:-1] + (-1,))

        return reconstruction.transpose()
