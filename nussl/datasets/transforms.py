from .. import AudioSignal
import numpy as np

def compute_ideal_binary_mask(source_magnitudes):
    ibm = (
        source_magnitudes == np.max(source_magnitudes, axis=-1, keepdims=True)
    ).astype(float)

    ibm = ibm / np.sum(ibm, axis=-1, keepdims=True)

    return ibm

class MagnitudeSpectrumApproximation(object):
    """
    Takes a dictionary and looks for two special keys, defined by the
    arguments ``mix_key`` and ``source_key``. These default to `mix` and `sources`.
    These values of these keys are used to calculate the magnitude spectrum 
    approximation [1]. The input dictionary is modified to have additional
    keys:

    - mix_magnitude: The magnitude spectrogram of the mixture audio signal.
    - source_magnitudes: The magnitude spectrograms of each source spectrogram.
    - assignments: The ideal binary assignments for each time-frequency bin.

    [1] Erdogan, Hakan, John R. Hershey, Shinji Watanabe, and Jonathan Le Roux. 
        "Phase-sensitive and recognition-boosted speech separation using 
        deep recurrent neural networks." In 2015 IEEE International Conference 
        on Acoustics, Speech and Signal Processing (ICASSP), pp. 708-712. IEEE, 
        2015.
    
    Args:
        mix_key (str, optional): The key to look for in data for the mixture AudioSignal. 
            Defaults to 'mix'.
        source_key (str, optional): The key to look for in the data containing the list of
            source AudioSignals. Defaults to 'sources'.
        stft_params ([type], optional): The STFT Parameters to use for each AudioSignal
            object. Defaults to None.
    
    Raises:
            TransformException: if the expected keys are not in the dictionary, an
                Exception is raised.
        
    Returns:
        data: Modified version of the input dictionary.
    """

    def __init__(self, mix_key='mix', source_key='sources', stft_params=None):
        self.stft_params = stft_params
        self.mix_key = mix_key
        self.source_key = source_key

    def __call__(self, data):        
        if self.mix_key not in data or self.source_key not in data:
            raise TransformException(
                f"Expected {self.mix_key} and {self.source_key} in dictionary "
                f"passed to this Transform! Got {list(data.keys())}"
            )

        mixture = data[self.mix_key]
        sources = data[self.source_key]

        mixture.stft_params = self.stft_params
        mixture.stft()
        mix_magnitude = mixture.magnitude_spectrogram_data

        source_magnitudes = []
        for s in sources:
            s.stft_params = self.stft_params
            s.stft()
            source_magnitudes.append(s.magnitude_spectrogram_data)

        source_magnitudes = np.stack(source_magnitudes, axis=-1)

        source_magnitudes = np.minimum(
            mix_magnitude[..., None], source_magnitudes)
        
        data['ideal_binary_mask'] = compute_ideal_binary_mask(source_magnitudes)
        data['mix_magnitude'] = mix_magnitude
        data['source_magnitudes'] = source_magnitudes
        return data

class PhaseSensitiveSpectrumApproximation(object):
    """
    Takes a dictionary and looks for two special keys, defined by the
    arguments ``mix_key`` and ``source_key``. These default to `mix` and `sources`.
    These values of these keys are used to calculate the phase sensitive spectrum 
    approximation [1]. The input dictionary is modified to have additional
    keys:

    - mix_magnitude: The magnitude spectrogram of the mixture audio signal.
    - source_magnitudes: The magnitude spectrograms of each source spectrogram.
    - assignments: The ideal binary assignments for each time-frequency bin.

    [1] Erdogan, Hakan, John R. Hershey, Shinji Watanabe, and Jonathan Le Roux. 
        "Phase-sensitive and recognition-boosted speech separation using 
        deep recurrent neural networks." In 2015 IEEE International Conference 
        on Acoustics, Speech and Signal Processing (ICASSP), pp. 708-712. IEEE, 
        2015.
    
    Args:
        mix_key (str, optional): The key to look for in data for the mixture AudioSignal. 
            Defaults to 'mix'.
        source_key (str, optional): The key to look for in the data containing the list of
            source AudioSignals. Defaults to 'sources'.
        stft_params ([type], optional): The STFT Parameters to use for each AudioSignal
            object. Defaults to None.
    
    Raises:
            TransformException: if the expected keys are not in the dictionary, an
                Exception is raised.
        
    Returns:
        data: Modified version of the input dictionary.
    """

    def __init__(self, mix_key='mix', source_key='sources', stft_params=None):
        self.stft_params = stft_params
        self.mix_key = mix_key
        self.source_key = source_key

    def __call__(self, data):
        if self.mix_key not in data or self.source_key not in data:
            raise TransformException(
                f"Expected {self.mix_key} and {self.source_key} in dictionary "
                f"passed to this Transform! Got {list(data.keys())}"
            )
        
        mixture = data[self.mix_key]
        sources = data[self.source_key]

        mixture.stft_params = self.stft_params
        mix_stft = mixture.stft()
        mix_magnitude = np.abs(mix_stft)
        mix_angle = np.angle(mix_stft)

        source_angles = []
        source_magnitudes = []
        for s in sources:
            s.stft_params = self.stft_params
            _stft = s.stft()
            source_magnitudes.append(np.abs(_stft))
            source_angles.append(np.angle(_stft))

        source_magnitudes = np.stack(source_magnitudes, axis=-1)
        source_angles = np.stack(source_angles, axis=-1)

        source_magnitudes = np.maximum(
            0,
            np.minimum(
                mix_magnitude[..., None],
                source_magnitudes * np.cos(source_angles - mix_angle[..., None])
            )
        )

        assignments = (
            source_magnitudes == np.max(source_magnitudes, axis=-1, keepdims=True)
        ).astype(float)

        assignments = (
            assignments / 
            np.sum(assignments, axis=-1, keepdims=True)
        )
        
        data['ideal_binary_mask'] = compute_ideal_binary_mask(source_magnitudes)
        data['mix_magnitude'] = mix_magnitude
        data['source_magnitudes'] = source_magnitudes
        return data    


class Compose(object):
    """Composes several transforms together. Copied from torchvision implementation.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class TransformException(Exception):
    """
    Exception class for errors when working with transforms in nussl.
    """
    pass
