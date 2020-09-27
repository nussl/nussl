import os
import shutil
import logging
import random
from collections import OrderedDict

import torch
import zarr
import numcodecs
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from .. import utils

# This is for when you're running multiple
# training threads
if hasattr(numcodecs, 'blosc'):
    numcodecs.blosc.use_threads = False


def compute_ideal_binary_mask(source_magnitudes):
    ibm = (
            source_magnitudes == np.max(source_magnitudes, axis=-1, keepdims=True)
    ).astype(float)

    ibm = ibm / np.sum(ibm, axis=-1, keepdims=True)
    ibm[ibm <= .5] = 0
    return ibm


# Keys that correspond to the time-frequency representations after being passed through
# the transforms here.
time_frequency_keys = ['mix_magnitude', 'source_magnitudes', 'ideal_binary_mask', 'weights']


class SumSources(object):
    """
    Sums sources together. Looks for sources in ``data[self.source_key]``. If 
    a source belongs to a group, it is popped from the ``data[self.source_key]`` and
    summed with the other sources in the group. If there is a corresponding 
    group_name in group_names, it is named that in ``data[self.source_key]``. If
    group_names are not given, then the names are constructed using the keys
    in each group (e.g. `drums+bass+other`).

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
        >>> import nussl
        >>> tfm = nussl.datasets.transforms.SumSources(
                groupings=[['drums', 'bass', 'other]],
                group_names=['accompaniment],
            )
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
            group_names = ['+'.join(groupings[i]) for i in range(len(groupings))]
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


class LabelsToOneHot(object):
    """
    Takes a data dictionary with sources and their keys and converts the keys to
    a one-hot numpy array using the list in data['metadata']['labels'] to figure
    out which index goes where.
    """

    def __init__(self, source_key='sources'):
        self.source_key = source_key

    def __repr__(self):
        return f'{self.__class__.__name__}(source_key = {self.source_key})'

    def __call__(self, data):
        if 'metadata' not in data:
            raise TransformException(
                f"Expected metadata in data, got {list(data.keys())}")
        if 'labels' not in data['metadata']:
            raise TransformException(
                f"Expected labels in data['metadata'], got "
                f"{list(data['metadata'].keys())}")

        enc = OneHotEncoder(categories=[data['metadata']['labels']])

        sources = data[self.source_key]
        source_keys = [k.split('::')[0] for k in list(sources.keys())]
        source_labels = [[l] for l in sorted(source_keys)]

        one_hot_labels = enc.fit_transform(source_labels)
        data['one_hot_labels'] = one_hot_labels.toarray()

        return data


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

    This transform uses the STFTParams that are attached to the AudioSignal objects
    contained in ``data[mix_key]`` and ``data[source_key]``.

    [1] Erdogan, Hakan, John R. Hershey, Shinji Watanabe, and Jonathan Le Roux. 
        "Phase-sensitive and recognition-boosted speech separation using 
        deep recurrent neural networks." In 2015 IEEE International Conference 
        on Acoustics, Speech and Signal Processing (ICASSP), pp. 708-712. IEEE, 
        2015.
    
    Args:
        mix_key (str, optional): The key to look for in data for the mixture AudioSignal. 
          Defaults to 'mix'.
        source_key (str, optional): The key to look for in the data containing the dict of
          source AudioSignals. Defaults to 'sources'.
    
    Raises:
            TransformException: if the expected keys are not in the dictionary, an
              Exception is raised.
        
    Returns:
        data: Modified version of the input dictionary.
    """

    def __init__(self, mix_key='mix', source_key='sources'):
        self.mix_key = mix_key
        self.source_key = source_key

    def __call__(self, data):
        if self.mix_key not in data:
            raise TransformException(
                f"Expected {self.mix_key} in dictionary "
                f"passed to this Transform! Got {list(data.keys())}."
            )

        mixture = data[self.mix_key]
        mixture.stft()
        mix_magnitude = mixture.magnitude_spectrogram_data

        data['mix_magnitude'] = mix_magnitude

        if self.source_key not in data:
            return data

        _sources = data[self.source_key]
        source_names = sorted(list(_sources.keys()))

        sources = OrderedDict()
        for key in source_names:
            sources[key] = _sources[key]
        data[self.source_key] = sources

        source_magnitudes = []
        for key in source_names:
            s = sources[key]
            s.stft()
            source_magnitudes.append(s.magnitude_spectrogram_data)

        source_magnitudes = np.stack(source_magnitudes, axis=-1)

        data['ideal_binary_mask'] = compute_ideal_binary_mask(source_magnitudes)
        data['source_magnitudes'] = source_magnitudes

        return data

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"mix_key = {self.mix_key}, "
            f"source_key = {self.source_key}"
            f")"
        )


class MagnitudeWeights(object):
    """
    Applying time-frequency weights to the deep clustering objective results in a
    huge performance boost. This transform looks for 'mix_magnitude', which is output
    by either MagnitudeSpectrumApproximation or PhaseSensitiveSpectrumApproximation
    and puts it into the weights.

    [1] Wang, Zhong-Qiu, Jonathan Le Roux, and John R. Hershey. 
    "Alternative objective functions for deep clustering." 2018 IEEE International 
    Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2018.
    
    Args:
        mix_magnitude_key (str): Which key to look for the mix_magnitude data in.
    """

    def __init__(self, mix_key='mix', mix_magnitude_key='mix_magnitude'):
        self.mix_magnitude_key = mix_magnitude_key
        self.mix_key = mix_key

    def __repr__(self):
        return f'{self.__class__.__name__}(mix_key = {self.mix_key}, ' \
               f'mix_magnitude_key = {self.mix_magnitude_key})'

    def __call__(self, data):
        if self.mix_magnitude_key not in data and self.mix_key not in data:
            raise TransformException(
                f"Expected {self.mix_magnitude_key} or {self.mix_key} in dictionary "
                f"passed to this Transform! Got {list(data.keys())}. "
                "Either MagnitudeSpectrumApproximation or "
                "PhaseSensitiveSpectrumApproximation should be called "
                "on the data dict prior to this transform. "
            )
        elif self.mix_magnitude_key not in data:
            data[self.mix_magnitude_key] = np.abs(data[self.mix_key].stft())

        magnitude_spectrogram = data[self.mix_magnitude_key]
        weights = magnitude_spectrogram / (np.sum(magnitude_spectrogram) + 1e-6)
        weights *= (
                magnitude_spectrogram.shape[0] * magnitude_spectrogram.shape[1]
        )
        data['weights'] = np.sqrt(weights)
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

    ``data[self.source_key]`` points to a dictionary containing the source names in
    the keys and the corresponding AudioSignal in the values. The keys are sorted
    in alphabetical order and then appended to the mask. ``data[self.source_key]``
    then points to an OrderedDict instead, where the keys are in the same order
    as in ``data['source_magnitudes']`` and ``data['assignments']``.

    This transform uses the STFTParams that are attached to the AudioSignal objects
    contained in ``data[mix_key]`` and ``data[source_key]``.

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
        range_min (float, optional): The lower end to use when truncating the source 
          magnitudes in the phase sensitive spectrum approximation. Defaults to 0.0 (construct
          non-negative masks). Use -np.inf for untruncated source magnitudes.
        range_max (float, optional): The higher end of the truncated spectrum. This gets
          multiplied by the magnitude of the mixture. Use 1.0 to truncate the source 
          magnitudes to `max(source_magnitudes, mix_magnitude)`. Use np.inf for untruncated
          source magnitudes (best performance for an oracle mask but may be beyond what a
          neural network is capable of masking). Defaults to 1.0.
          
    Raises:
            TransformException: if the expected keys are not in the dictionary, an
              Exception is raised.
        
    Returns:
        data: Modified version of the input dictionary.
    """

    def __init__(self, mix_key='mix', source_key='sources',
                 range_min=0.0, range_max=1.0):
        self.mix_key = mix_key
        self.source_key = source_key
        self.range_min = range_min
        self.range_max = range_max

    def __call__(self, data):
        if self.mix_key not in data:
            raise TransformException(
                f"Expected {self.mix_key} in dictionary "
                f"passed to this Transform! Got {list(data.keys())}."
            )

        mixture = data[self.mix_key]

        mix_stft = mixture.stft()
        mix_magnitude = np.abs(mix_stft)
        mix_angle = np.angle(mix_stft)
        data['mix_magnitude'] = mix_magnitude

        if self.source_key not in data:
            return data

        _sources = data[self.source_key]
        source_names = sorted(list(_sources.keys()))

        sources = OrderedDict()
        for key in source_names:
            sources[key] = _sources[key]
        data[self.source_key] = sources

        source_angles = []
        source_magnitudes = []
        for key in source_names:
            s = sources[key]
            _stft = s.stft()
            source_magnitudes.append(np.abs(_stft))
            source_angles.append(np.angle(_stft))

        source_magnitudes = np.stack(source_magnitudes, axis=-1)
        source_angles = np.stack(source_angles, axis=-1)
        range_min = self.range_min
        range_max = self.range_max * mix_magnitude[..., None]

        # Section 3.1: https://arxiv.org/pdf/1909.08494.pdf
        source_magnitudes = np.minimum(
            np.maximum(
                source_magnitudes * np.cos(source_angles - mix_angle[..., None]),
                range_min
            ),
            range_max
        )

        data['ideal_binary_mask'] = compute_ideal_binary_mask(source_magnitudes)
        data['source_magnitudes'] = source_magnitudes

        return data

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"mix_key = {self.mix_key}, "
            f"source_key = {self.source_key}, "
            f"range_min = {self.range_min}, "
            f"range_max = {self.range_max}"
            f")"
        )


class IndexSources(object):
    """
    Takes in a dictionary containing Torch tensors or numpy arrays and extracts the 
    indexed sources from the set key (usually either `source_magnitudes` or 
    `ideal_binary_mask`). Can be used to train single-source separation models 
    (e.g. mix goes in, vocals come out).
    
    You need to know which slice of the source magnitudes or ideal binary mask arrays 
    to extract. The order of the sources in the source magnitudes array will be in 
    alphabetical order according to their source labels. 

    For example, if source magnitudes has shape `(257, 400, 1, 4)`, and the data is
    from MUSDB, then the four possible source labels are bass, drums, other, and vocals.
    The data in source magnitudes is in alphabetical order, so:

    .. code-block:: python

        # source_magnitudes is an array returned by either MagnitudeSpectrumApproximation
        # or PhaseSensitiveSpectrumApproximation
        source_magnitudes[..., 0] # bass spectrogram
        source_magnitudes[..., 1] # drums spectrogram
        source_magnitudes[..., 2] # other spectrogram
        source_magnitudes[..., 3] # vocals spectrogram

        # ideal_binary_mask is an array returned by either MagnitudeSpectrumApproximation
        # or PhaseSensitiveSpectrumApproximation
        ideal_binary_mask[..., 0] # bass ibm mask
        ideal_binary_mask[..., 1] # drums ibm mask
        ideal_binary_mask[..., 2] # other ibm mask
        ideal_binary_mask[..., 3] # vocals ibm mask

    You can apply this transform to either the `source_magnitudes` or the
    `ideal_binary_mask` or both.

    
    Args:
        object ([type]): [description]
    """

    def __init__(self, target_key, index):
        self.target_key = target_key
        self.index = index

    def __call__(self, data):
        if self.target_key not in data:
            raise TransformException(
                f"Expected {self.target_key} in dictionary, got {list(data.keys())}")
        if self.index >= data[self.target_key].shape[-1]:
            raise TransformException(
                f"Shape of data[{self.target_key}] is {data[self.target_key].shape} "
                f"but index =  {self.index} out of bounds bounds of last dim.")
        data[self.target_key] = data[self.target_key][..., self.index, None]
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(target_key = {self.target_key}, ' \
               f'index = {self.index})'


class GetExcerpt(object):
    """
    Takes in a dictionary containing Torch tensors or numpy arrays and extracts an 
    excerpt from each tensor corresponding to a spectral representation of a specified 
    length in frames. Can be used to get L-length spectrograms from mixture and source
    spectrograms. If the data is shorter than the specified length, it
    is padded to the specified length. If it is longer, a random offset between
    ``(0, data_length - specified_length)`` is chosen. This function assumes that
    it is being passed data AFTER ToSeparationModel. Thus the time dimension is
    on axis=1. 
    
    Args:
        excerpt_length (int): Specified length of transformed data in frames.

        time_dim (int): Which dimension time is on (excerpts are taken along this axis).
          Defaults to 0.

        time_frequency_keys (list): Which keys to look at it in the data dictionary to
          take excerpts from.
    """

    def __init__(self, excerpt_length, time_dim=0,
                 tf_keys=None):
        self.excerpt_length = excerpt_length
        self.time_dim = time_dim
        self.time_frequency_keys = tf_keys if tf_keys else time_frequency_keys

    def __repr__(self):
        return f'{self.__class__.__name__}(excerpt_length = {self.excerpt_length}), ' \
               f'time_dim = {self.time_dim}, tf_keys = {self.time_frequency_keys})'

    @staticmethod
    def _validate(data, key):
        is_tensor = torch.is_tensor(data[key])
        is_array = isinstance(data[key], np.ndarray)
        if not is_tensor and not is_array:
            raise TransformException(
                f"data[{key}] was not a torch Tensor or a numpy array!")
        return is_tensor, is_array

    def _get_offset(self, data, key):
        self._validate(data, key)
        data_length = data[key].shape[self.time_dim]

        if data_length >= self.excerpt_length:
            offset = random.randint(0, data_length - self.excerpt_length)
        else:
            offset = 0

        pad_amount = max(0, self.excerpt_length - data_length)

        return offset, pad_amount

    def _construct_pad_func_tuple(self, shape, pad_amount, is_tensor):
        if is_tensor:
            pad_func = torch.nn.functional.pad
            pad_tuple = [0 for _ in range(2 * len(shape))]
            pad_tuple[2 * self.time_dim] = pad_amount
            pad_tuple = pad_tuple[::-1]
        else:
            pad_func = np.pad
            pad_tuple = [(0, 0) for _ in range(len(shape))]
            pad_tuple[self.time_dim] = (0, pad_amount)
        return pad_func, pad_tuple

    def __call__(self, data):
        offset, pad_amount = self._get_offset(
            data, self.time_frequency_keys[0])

        for key in data:
            if key in self.time_frequency_keys:
                is_tensor, is_array = self._validate(data, key)

                if pad_amount > 0:
                    pad_func, pad_tuple = self._construct_pad_func_tuple(
                        data[key].shape, pad_amount, is_tensor)
                    data[key] = pad_func(data[key], pad_tuple)

                data[key] = utils._slice_along_dim(
                    data[key], self.time_dim, offset, offset + self.excerpt_length)
        return data


class Cache(object):
    """
    The Cache transform can be placed within a Compose transform. The data 
    dictionary coming into this transform will be saved to the specified
    location using ``zarr``. Then instead of computing all of the transforms
    before the cache, one can simply read from the cache. The transforms after
    this will then be applied to the data dictionary that is read from the
    cache. A typical pipeline might look like this:

    .. code-block:: python

        dataset = datasets.Scaper('path/to/scaper/folder')
        tfm = transforms.Compose([
            transforms.PhaseSensitiveApproximation(),
            transforms.ToSeparationModel(),
            transforms.Cache('~/.nussl/cache/tag', overwrite=True),
            transforms.GetExcerpt()
        ])
        dataset[0] # first time will write to cache then apply GetExcerpt
        dataset.cache_populated = True # switches to reading from cache
        dataset[0] # second time will read from cache then apply GetExcerpt
        dataset[1] # will error out as it wasn't written to the cache!

        dataset.cache_populated = False
        for i in range(len(dataset)):
            dataset[i] # every item will get written to cache
        dataset.cache_populated = True
        dataset[1] # now it exists

        dataset = datasets.Scaper('path/to/scaper/folder') # next time around
        tfm = transforms.Compose([
            transforms.PhaseSensitiveApproximation(),
            transforms.ToSeparationModel(),
            transforms.Cache('~/.nussl/cache/tag', overwrite=False),
            transforms.GetExcerpt()
        ])
        dataset.cache_populated = True
        dataset[0] # will read from cache, which still exists from last time
    
    Args:
        object ([type]): [description]
    """

    def __init__(self, location, cache_size=1, overwrite=False):
        self.location = location
        self.cache_size = cache_size
        self.cache = None
        self.overwrite = overwrite

    @property
    def info(self):
        return self.cache.info

    @property
    def overwrite(self):
        return self._overwrite

    @overwrite.setter
    def overwrite(self, value):
        self._overwrite = value
        self._clear_cache(self.location)
        self._open_cache(self.location)

    def _clear_cache(self, location):
        if os.path.exists(location):
            if self.overwrite:
                logging.info(
                    f"Cache {location} exists and overwrite = True, clearing cache.")
                shutil.rmtree(location, ignore_errors=True)

    def _open_cache(self, location):
        if self.overwrite:
            self.cache = zarr.open(location, mode='w', shape=(self.cache_size,),
                                   chunks=(1,), dtype=object, 
                                   object_codec=numcodecs.Pickle(),
                                   synchronizer=zarr.ThreadSynchronizer())
        else:
            if os.path.exists(location):
                self.cache = zarr.open(location, mode='r',
                    object_codec=numcodecs.Pickle(),
                    synchronizer=zarr.ThreadSynchronizer())

    def __call__(self, data):
        if 'index' not in data:
            raise TransformException(
                f"Expected 'index' in dictionary, got {list(data.keys())}")
        index = data['index']
        if self.overwrite:
            self.cache[index] = data
        data = self.cache[index]

        if not isinstance(data, dict):
            raise TransformException(
                f"Reading from cache resulted in not a dictionary! "
                f"Maybe you haven't written to index {index} yet in "
                f"the cache?")

        return data


class GetAudio(object):
    """
    Extracts the audio from each signal in `mix_key` and `source_key`. 
    These will be at new keys, called `mix_audio` and `source_audio`.
    Can be used for training end-to-end models.
    
    Args:
        mix_key (str, optional): The key to look for in data for the mixture AudioSignal. 
          Defaults to 'mix'.
        source_key (str, optional): The key to look for in the data containing the dict of
          source AudioSignals. Defaults to 'sources'.
    """
    def __init__(self, mix_key='mix', source_key='sources'):
        self.mix_key = mix_key
        self.source_key = source_key

    def __repr__(self):
        return f'{self.__class__.__name__}(mix_key = {self.mix_key}, ' \
               f'source_key = {self.source_key})'

    def __call__(self, data):
        if self.mix_key not in data:
            raise TransformException(
                f"Expected {self.mix_key} in dictionary "
                f"passed to this Transform! Got {list(data.keys())}."
            )
        
        mix = data[self.mix_key]
        data['mix_audio'] = mix.audio_data

        if self.source_key not in data:
            return data

        _sources = data[self.source_key]
        source_names = sorted(list(_sources.keys()))
        
        source_audio = []
        for key in source_names:
            source_audio.append(_sources[key].audio_data)
        # sources on last axis
        source_audio = np.stack(source_audio, axis=-1)

        data['source_audio'] = source_audio
        return data


class ToSeparationModel(object):
    """
    Takes in a dictionary containing objects and removes any objects that cannot
    be passed to SeparationModel (e.g. not a numpy array or torch Tensor).
    If these objects are passed to SeparationModel, then an error will occur. This 
    class should be the last one in your list of transforms, if you're using 
    this dataset in a DataLoader object for training a network. If the keys
    correspond to numpy arrays, they are converted to tensors using 
    ``torch.from_numpy``. Finally, the dimensions corresponding to time and
    frequency are swapped for all the keys in swap_tf_dims, as this is how 
    SeparationModel expects it.

    Example:

    .. code-block:: none

        data = {
            # 2ch spectrogram for mixture
            'mix_magnitude': torch.randn(513, 400, 2),
            # 2ch spectrogram for each source
            'source_magnitudes': torch.randn(513, 400, 2, 4) 
            'mix': AudioSignal()
        }

        tfm = transforms.ToSeparationModel()
        data = tfm(data)

        data['mix_magnitude'].shape # (400, 513, 2)
        data['source_magnitudes].shape # (400, 513, 2, 4)
        'mix' in data.keys() # False
    

    If this class isn't in your transforms list for the dataset, but you are
    using it in the Trainer class, then it is added automatically as the
    last transform.
    """

    def __init__(self, swap_tf_dims=None):
        self.swap_tf_dims = swap_tf_dims if swap_tf_dims else time_frequency_keys

    def __call__(self, data):
        keys = list(data.keys())
        for key in keys:
            if key != 'index':
                is_array = isinstance(data[key], np.ndarray)
                if is_array:
                    data[key] = torch.from_numpy(data[key])
                if not torch.is_tensor(data[key]):
                    data.pop(key)
                if key in self.swap_tf_dims:
                    data[key] = data[key].transpose(1, 0)
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(swap_tf_dims = {self.swap_tf_dims})"


class Compose(object):
    """Composes several transforms together. Inspired by torchvision implementation.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.MagnitudeSpectrumApproximation(),
        >>>     transforms.ToSeparationModel(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if not isinstance(data, dict):
                raise TransformException(
                    "The output of every transform must be a dictionary!")
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



