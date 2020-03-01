from .. import AudioSignal
import numpy as np
from collections import OrderedDict
import torch

def compute_ideal_binary_mask(source_magnitudes):
    ibm = (
        source_magnitudes == np.max(source_magnitudes, axis=-1, keepdims=True)
    ).astype(float)

    ibm = ibm / np.sum(ibm, axis=-1, keepdims=True)
    return ibm

class SumSources(object):
    """
    Sums sources together. Looks for sources in ``data[self.source_key]``. If 
    a source belongs to a group, it is popped from the ``data[self.source_key]`` and
    summed with the other sources in the group. If there is a corresponding 
    group_name in group_names, it is named that in ``data[self.source_key]``. If
    group_names are not given, then the names are ``['group0', 'group1', ...]``.

    If using Scaper datasets, then there may be multiple sources with the same
    label but different counts. The Scaper dataset hook organizes the source
    dictionary as follows:

    .. code-block:: none

        data['sources] = {
            '{label}::{count}': AudioSignal,
            '{label}::{count}': AudioSignal,
            ...
        }
    
    SumSources sums by source label, so the ``::count`` will be ignored and only the
    label part will be used when grouping sources.

    Example:
        >>> tfm = transforms.SumSources(
        >>>     groupings=[['drums', 'bass', 'other]],
        >>>     group_names=['accompaniment],
        >>> )
        >>> # data['sources'] is a dict containing keys: 
        >>> #   ['vocals', 'drums', 'bass', 'other]
        >>> data = tfm(data)
        >>> # data['sources'] is now a dict containing keys:
        >>> #   ['vocals', 'accompaniment']
    
    Args:
        groupings (list): a list of lists telling how to group each sources. 
        group_names (list, optional): A list containing the names of each group, or None. 
            Defaults to None.
        source_key (str, optional): The key to look for in the data containing the list of
            source AudioSignals. Defaults to 'sources'.
    
    Raises:
        TransformException: if groupings is not a list
        TransformException: if group_names is not None but 
            len(groupings) != len(group_names)
    
    Returns:
        data: modified dictionary with summed sources
    """

    def __init__(self, groupings, group_names=None, source_key='sources'):
        if not isinstance(groupings, list):
            raise TransformException(
                f"groupings must be a list, got {type(groupings)}!")
        
        if group_names:
            if len(group_names) != len(groupings):
                raise TransformException(
                    f"group_names and groupings must be same length or "
                    f"group_names can be None! Got {len(group_names)} for "
                    f"len(group_names) and {len(groupings)} for len(groupings)."
                )
        
        self.groupings = groupings
        self.source_key = source_key
        if group_names is None:
            group_names = [f"group{i}" for i in range(len(groupings))]
        self.group_names = group_names
    
    def __call__(self, data):
        if self.source_key not in data:
            raise TransformException(
                f"Expected {self.source_key} in dictionary "
                f"passed to this Transform!"
            )
        sources = data[self.source_key]
        source_keys = [(k.split('::')[0], k) for k in list(sources.keys())]

        for i, group in enumerate(self.groupings):
            combined = []
            group_name = self.group_names[i]
            for key1 in group:
                for key2 in source_keys:
                    if key2[0] == key1:
                        combined.append(sources[key2[1]])
                        sources.pop(key2[1])
            sources[group_name] = sum(combined)
            sources[group_name].path_to_input_file = group_name
        
        data[self.source_key] = sources
        if 'metadata' in data:
            if 'labels' in data['metadata']:
                data['metadata']['labels'].extend(self.group_names)

        return data

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"groupings = {self.groupings}, "
            f"group_names = {self.group_names}, "
            f"source_key = {self.source_key}"
            f")"
        )

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

    ``data[self.source_key]`` points to a dictionary containing the source names in
    the keys and the corresponding AudioSignal in the values. The keys are sorted
    in alphabetical order and then appended to the mask. ``data[self.source_key]``
    then points to an OrderedDict instead, where the keys are in the same order
    as in ``data['source_magnitudes']`` and ``data['assignments']``.

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
                f"passed to this Transform! Got {list(data.keys())}."
            )

        mixture = data[self.mix_key]
        _sources = data[self.source_key]
        source_names = sorted(list(_sources.keys()))

        sources = OrderedDict()
        for key in source_names:
            sources[key] = _sources[key]
        data[self.source_key] = sources

        mixture.stft_params = self.stft_params
        mixture.stft()
        mix_magnitude = mixture.magnitude_spectrogram_data

        source_magnitudes = []
        for key in source_names:
            s = sources[key]
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

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"stft_params = {self.stft_params}, "
            f"mix_key = {self.mix_key}, "
            f"source_key = {self.source_key}"
            f")"
        )

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

    ``data[self.source_key]`` points to a dictionary containing the source names in
    the keys and the corresponding AudioSignal in the values. The keys are sorted
    in alphabetical order and then appended to the mask. ``data[self.source_key]``
    then points to an OrderedDict instead, where the keys are in the same order
    as in ``data['source_magnitudes']`` and ``data['assignments']``.

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
                f"passed to this Transform! Got {list(data.keys())}."
            )
        
        mixture = data[self.mix_key]
        _sources = data[self.source_key]
        source_names = sorted(list(_sources.keys()))

        sources = OrderedDict()
        for key in source_names:
            sources[key] = _sources[key]
        data[self.source_key] = sources

        mixture.stft_params = self.stft_params
        mix_stft = mixture.stft()
        mix_magnitude = np.abs(mix_stft)
        mix_angle = np.angle(mix_stft)

        source_angles = []
        source_magnitudes = []
        for key in source_names:
            s = sources[key]
            s.stft_params = self.stft_params
            _stft = s.stft()
            source_magnitudes.append(np.abs(_stft))
            source_angles.append(np.angle(_stft))

        source_magnitudes = np.stack(source_magnitudes, axis=-1)
        source_angles = np.stack(source_angles, axis=-1)

        # Section 3.1: https://arxiv.org/pdf/1909.08494.pdf
        source_magnitudes = np.minimum(
            np.maximum(
                source_magnitudes * np.cos(source_angles - mix_angle[..., None]),
                0
            ),
            mix_magnitude[..., None]
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

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"stft_params = {self.stft_params}, "
            f"mix_key = {self.mix_key}, "
            f"source_key = {self.source_key}"
            f")"
        )

class ToDataLoader(object):
    """
    Takes in a dictionary containing objects and removes any objects that cannot
    be passed to ``torch.datasets.DataLoader`` (e.g. not a numpy array or torch Tensor).
    If these objects are passed to a DataLoader, then an error will occur. This 
    class should be the last one in your list of transforms, if you're using 
    this dataset in a DataLoader object for training a network.

    If this class isn't in your transforms list for the dataset, but you are
    using it in the Trainer class, then it is added automatically as the
    last transform.
    """
    def __init__(self):
        pass

    def __call__(self, data):
        keys = list(data.keys())
        for key in keys:
            is_array = isinstance(data[key], np.ndarray)
            is_tensor = torch.is_tensor(data[key])
            if not is_tensor and not is_array:
                data.pop(key)
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}()"

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
