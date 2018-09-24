#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Provides utilities for running nussl algorithms that do not belong to
any specific algorithm or that are shared between algorithms.
"""

from __future__ import division
import warnings
import base64
import json
import re
import collections

import numpy as np
import musdb

import constants

__all__ = ['find_peak_indices', 'find_peak_values',
           'json_ready_numpy_array', 'json_serialize_numpy_array', 'load_numpy_json',
           'json_numpy_obj_hook',
           'add_mismatched_arrays', 'add_mismatched_arrays2D', 'complex_randn',
           '_get_axis',
           'print_all_separation_algorithms',
           'verify_audio_signal_list_lax', 'verify_audio_signal_list_strict',
           'verify_separation_base_list', 'verify_mask_separation_base_list',
           '_verify_audio_data', '_verify_transformation_data']


def find_peak_indices(input_array, n_peaks, min_dist=None, do_min=False, threshold=0.5):
    """
    This function will find the indices of the peaks of an input n-dimensional numpy array.
    This can be configured to find max or min peak indices, distance between the peaks, and
    a lower bound, at which the algorithm will stop searching for peaks (or upper bound if
    searching for max). Used exactly the same as :func:`find_peak_values`.

    This function currently only accepts 1-D and 2-D numpy arrays.

    Notes:
        * This function only returns the indices of peaks. If you want to find peak values,
        use :func:`find_peak_values`.

        * min_dist can be an int or a tuple of length 2.
            If input_array is 1-D, min_dist must be an integer.
            If input_array is 2-D, min_dist can be an integer, in which case the minimum
            distance in both dimensions will be equal. min_dist can also be a tuple if
            you want each dimension to have a different minimum distance between peaks.
            In that case, the 0th value in the tuple represents the first dimension, and
            the 1st value represents the second dimension in the numpy array.


    See Also:
        :: :func:`find_peak_values` ::

    Args:
        input_array: a 1- or 2- dimensional numpy array that will be inspected.
        n_peaks: (int) maximum number of peaks to find
        min_dist: (int) minimum distance between peaks. Default value: len(input_array) / 4
        do_min: (bool) if True, finds indices at minimum value instead of maximum
        threshold: (float) the value (scaled between 0.0 and 1.0)

    Returns:
        peak_indices: (list) list of the indices of the peak values

    """
    input_array = np.array(input_array, dtype=float)

    if input_array.ndim > 2:
        raise ValueError('Cannot find peak indices on data greater than 2 dimensions!')

    is_1d = input_array.ndim == 1
    zero_dist = zero_dist0 = zero_dist1 = None
    min_dist = len(input_array) // 4 if min_dist is None else min_dist

    if is_1d:
        zero_dist = min_dist
    else:
        if type(min_dist) is int:
            zero_dist0 = zero_dist1 = min_dist
        elif len(min_dist) == 1:
            zero_dist0 = zero_dist1 = min_dist[0]
        else:
            zero_dist0, zero_dist1 = min_dist

    # scale input_array between [0.0, 1.0]
    if np.min(input_array) < 0.0:
        input_array += np.min(input_array)
    elif np.min(input_array) > 0.0:
        input_array -= np.min(input_array)

    input_array /= np.max(input_array)

    # flip sign if doing min
    input_array = -input_array if do_min else input_array

    # throw out everything below threshold
    input_array = np.multiply(input_array, (input_array >= threshold))

    # check to make sure we didn't throw everything out
    if np.size(np.nonzero(input_array)) == 0:
        raise ValueError('Threshold set incorrectly. No peaks above threshold.')
    if np.size(np.nonzero(input_array)) < n_peaks:
        warnings.warn('Threshold set such that there will be less peaks than n_peaks.')

    peak_indices = []
    for i in range(n_peaks):
        # np.unravel_index for 2D indices e.g., index 5 in a 3x3 array should be (1, 2)
        # Also, wrap in list for duck typing
        cur_peak_idx = list(np.unravel_index(np.argmax(input_array), input_array.shape))

        # zero out peak and its surroundings
        if is_1d:
            cur_peak_idx = cur_peak_idx[0]
            peak_indices.append(cur_peak_idx)
            lower, upper = _set_array_zero_indices(cur_peak_idx, zero_dist, len(input_array))
            input_array[lower:upper] = 0
        else:
            peak_indices.append(cur_peak_idx)
            lower0, upper0 = _set_array_zero_indices(cur_peak_idx[0], zero_dist0,
                                                     input_array.shape[0])
            lower1, upper1 = _set_array_zero_indices(cur_peak_idx[1], zero_dist1,
                                                     input_array.shape[1])
            input_array[lower0:upper0, lower1:upper1] = 0

        if np.sum(input_array) == 0.0:
            break

    return peak_indices


def _set_array_zero_indices(index, zero_distance, max_len):
    lower = index - zero_distance - 1
    upper = index + zero_distance + 1
    lower = 0 if lower < 0 else lower
    upper = max_len if upper >= max_len else upper
    return lower, upper


def find_peak_values(input_array, n_peaks, min_dist=None, do_min=False, threshold=0.5):
    """
    Finds the values of the peaks in a 1-D or 2-D numpy array. Used exactly the same as
    :func:`find_peak_indices`. This function will find the values of the peaks of an input
    n-dimensional numpy array.

    This can be configured to find max or min peak values, distance between the peaks, and
    a lower bound, at which the algorithm will stop searching for peaks (or upper bound if
    searching for max).

    This function currently only accepts 1-D and 2-D numpy arrays.

    Notes:
        * This function only returns the indices of peaks. If you want to find peak values,
        use :func:`find_peak_indices`.

        * min_dist can be an int or a tuple of length 2.
            If input_array is 1-D, min_dist must be an integer.
            If input_array is 2-D, min_dist can be an integer, in which case the minimum
            distance in both dimensions will be equal. min_dist can also be a tuple if
            you want each dimension to have a different minimum distance between peaks.
            In that case, the 0th value in the tuple represents the first dimension, and
            the 1st value represents the second dimension in the numpy array.


    See Also:
        :: :func:`find_peak_indices` ::

    Args:
        input_array: a 1- or 2- dimensional numpy array that will be inspected.
        n_peaks: (int) maximum number of peaks to find
        min_dist: (int) minimum distance between peaks. Default value: len(input_array) / 4
        do_min: (bool) if True, finds indices at minimum value instead of maximum
        threshold: (float) the value (scaled between 0.0 and 1.0)

    Returns:
        peak_values: (list) list of the values of the peak values

    """
    if input_array.ndim > 2:
        raise ValueError('Cannot find peak indices on data greater than 2 dimensions!')

    if input_array.ndim == 1:
        return [input_array[i] for i in find_peak_indices(input_array, n_peaks, min_dist,
                                                          do_min, threshold)]
    else:
        return [input_array[i, j] for i, j in find_peak_indices(input_array, n_peaks, min_dist,
                                                                do_min, threshold)]


def json_ready_numpy_array(array):
    """
    Adapted from:
    http://stackoverflow.com/a/27948073/5768001
    Args:
        array: np array to make json ready.

    Returns:

    """
    if isinstance(array, np.ndarray):
        # noinspection PyTypeChecker
        data_b64 = base64.b64encode(np.ascontiguousarray(array).data)
        return {
                constants.NUMPY_JSON_KEY: {
                        "__ndarray__": data_b64,
                        "dtype":  str(array.dtype),
                        "shape": array.shape
                    }
                }

    return None


def json_serialize_numpy_array(array):
    """
    Returns a JSON string of the numpy array.

    Notes:
        The generated JSON strings can be converted back to numpy arrays with load_numpy_json()

    Args:
        array: (numpy array) any numpy array to convert to JSON

    Returns:
        (string) JSON-ified numpy array.

    See Also:
        load_numpy_json()
    """
    return json.dumps(json_ready_numpy_array(array))


def load_numpy_json(array_json):
    """
    Turns a JSON-ified numpy array back into a regular numpy array.

    Notes:
        This function is only guaranteed to work with JSON generated by json_serialize_numpy_array()

    Args:
        array_json: (string) JSON-ified nump array

    Returns:
        (numpy array) numpy array from the input JSON string

    See Also:
        json_serialize_numpy_array()
    """
    return json.loads(array_json, object_hook=json_numpy_obj_hook)[constants.NUMPY_JSON_KEY]


def json_numpy_obj_hook(dct):
    """
    Decodes a previously encoded numpy ndarray
    with proper shape and dtype
    from: http://stackoverflow.com/a/27948073/5768001
    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct


def add_mismatched_arrays(array1, array2, truncate=False):
    """
    Will add two 1D numpy arrays of different length. If :param:`truncate` is ``False``, it will
    expand the resultant array to the larger of the two, if ``True`` it will truncate the resultant
    array to the smaller of the two.

    Args:
        array1: (:obj:`np.ndarray`) 1D numeric numpy array
        array2: (:obj:`np.ndarray`) 1D numeric numpy array
        truncate: (bool) If ``Tru``e, will truncate the resultant array to the smaller of the two

    Returns:
        (:obj:`np.ndarray`): A 1D numpy array that is the element-wise sum of the two input arrays.

    """
    # Cast these arrays to the largest common type
    array1 = np.array(array1, dtype=np.promote_types(array1.dtype, array2.dtype))
    array2 = np.array(array2, dtype=np.promote_types(array1.dtype, array2.dtype))

    # TODO: find a more elegant way to do this whole function

    if truncate:
        if len(array1) < len(array2):
            result = array1.copy()
            result += array2[:len(array1)]
        else:
            result = array2.copy()
            result += array1[:len(array2)]
    else:
        if len(array1) < len(array2):
            result = array2.copy()
            result[:len(array1)] += array1
        else:
            result = array1.copy()
            result[:len(array2)] += array2

    return result


# noinspection PyPep8Naming
def add_mismatched_arrays2D(array1, array2, truncate=False):
    """
    Will add two 2D numpy arrays of different length. If :param:`truncate` is ``False``, it will
    expand the resultant array to the larger of the two, if ``True`` it will truncate the resultant
    array to the smaller of the two.

    Args:
        array1: (:obj:`np.array`) 2D numeric numpy array.
        array2: (:obj:`np.array`) 2D numeric numpy array.
        truncate: (bool) If ``True``, will truncate the resultant array to the smaller of the two.

    Returns:
        (:obj:`np.ndarray`): A 2D numpy array that is the element-wise sum of the two input arrays.

    """
    # Cast these arrays to the largest common type
    array1 = np.array(array1, dtype=np.promote_types(array1.dtype, array2.dtype))
    array2 = np.array(array2, dtype=np.promote_types(array1.dtype, array2.dtype))

    # TODO: find a more elegant way to do this whole function

    if truncate:
        if array1.shape[1] < array2.shape[1]:  # Kludge
            result = array1.copy()
            result += array2[:, :array1.shape[1]]
        else:
            result = array2.copy()
            result += array1[:, :array2.shape[1]]
    else:
        if array1.shape[1] < array2.shape[1]:
            result = array2.copy()
            result[:, :array1.shape[1]] += array1
        else:
            result = array1.copy()
            result[:, :array2.shape[1]] += array2

    return result


def complex_randn(shape):
    """
    Returns a complex-valued numpy array of random values with shape :param:`shape`.

    Args:
        shape (tuple): Tuple of ints that will be the shape of the resultant complex numpy array.

    Returns:
        (:obj:`np.ndarray`): a complex-valued numpy array of random values with shape `shape`
    """
    return np.random.randn(*shape) + 1j * np.random.randn(*shape)


def _get_axis(array, axis_num, i):
    """
    Will get index 'i' along axis 'axis_num' of a 2- or 3-dimensional numpy array.
    If array has 4+ dimensions or 'axis_num' is larger than number of axes, will return None.

    Args:
        array (:obj:`np.ndarray`): Array to fetch axis of.
        axis_num (int): Axes to retrieve.
        i (int): Index to retrieve.

    Returns:
        The value at index :param:`i` along axis :param:`axis_num`
    """
    if array.ndim == 2:
        if axis_num == 0:
            return array[i, :]
        elif axis_num == 1:
            return array[:, i]
        else:
            return None
    elif array.ndim == 3:
        if axis_num == 0:
            return array[i, :, :]
        elif axis_num == 1:
            return array[:, i, :]
        elif axis_num == 2:
            return array[:, :, i]
        else:
            return None
    else:
        return None


def CamelCase_to_snake_case(text):
    """
    Convert a string from camel case (``StringsThatLookLikeThis``) to snake case
    (``strings_that_look_like_this``). This is useful for matching names of
    Args:
        text:

    Returns:

    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def _format(string):
    """ Formats a class name correctly for checking function and class names.
        Strips all non-alphanumeric chars and makes lowercase.
    """
    return str(filter(str.isalnum, string)).lower()


def audio_signals_to_musdb_track(mixture, sources_dict, targets_dict):
    """
    Converts :class:`AudioSignal` objects to ``musdb`` :class:`Track` objects that
    contain the mixture, the ground truth sources, and the targets for use with the ``mus_eval``
    implementation of BSS-Eval and ``musdb``.

    See Also:
        * More information on ``musdb``:  `Github<https://github.com/sigsep/sigsep-mus-db>`
            and `documentation<http://musdb.readthedocs.io/>`
        * More information on ``mus_eval``: `Github<https://github.com/sigsep/sigsep-mus-eval>`
            and `documentation<https://sigsep.github.io/sigsep-mus-eval/>`
        * :class:`BSSEvalV4` for ``nussl``'s interface to BSS-Eval v4.

    Examples:
        .. code-block:: python
            :linenos:
            import nussl
            signal = nussl.AudioSignal(nussl.efz_utils.download_audio_file('HistoryRepeating.wav'))

            repet = nussl.Repet(signal)
            repet.run()

            bg, fg = repet.make_audio_signals()

            src_dict = {'vocals': fg, 'accompaniment': bg}
            target = nussl.core.constants.STEM_TARGET_DICT
            track = nussl.utils.audio_signals_to_musdb_track(signal, src_dict, target)

    Args:
        mixture (:class:`AudioSignal`): The :class:`AudioSignal` object that contains the mixture.
        sources_dict (dict): Dictionary where the keys are the labels for the sources and values
            are the associated :class:`AudioSignal` objects.
        targets_dict (dict): Dictionary where the keys are the labels for the sources (as above)
            and the values are weights.

    Returns:
        (:obj:`musdb.Track`) populated as specified by inputs.
    """
    verify_audio_signal_list_strict(sources_dict.values() + [mixture])

    track = musdb.Track(mixture.file_name, is_wav=True, stem_id=0)
    track.audio = mixture.audio_data.T
    track.rate = mixture.sample_rate

    sources = {}
    i = 1
    for key, sig in list(sources_dict.items()):
        sources[key] = musdb.Source(name=key, stem_id=i, is_wav=True)
        sources[key].audio = sig.audio_data.T
        sources[key].rate = sig.sample_rate
        i += 1

    track.sources = sources

    targets = collections.OrderedDict()
    for name, target_srcs in list(targets_dict.items()):
        # add a list of target sources
        target_sources = []
        for source, gain in list(target_srcs.items()):
            if source in list(track.sources.keys()):
                # add gain to source tracks
                track.sources[source].gain = float(gain)
                # add tracks to components
                target_sources.append(sources[source])
        # add sources to target
        if target_sources:
            targets[name] = musdb.Target(sources=target_sources)

    track.targets = targets
    return track


def verify_audio_signal_list_lax(audio_signal_list):
    """
    Verifies that an input (:param:`audio_signal_list`) is a list of :ref:`AudioSignal` objects.
    If not so, attempts to correct the list (if possible) and returns the corrected list.

    Args:
        audio_signal_list (list): List of :ref:`AudioSignal` objects

    Returns:
        audio_signal_list (list): Verified list of :ref:`AudioSignal` objects.

    """
    # Lazy load to prevent a circular reference upon initialization
    from .audio_signal import AudioSignal

    if isinstance(audio_signal_list, AudioSignal):
        audio_signal_list = [audio_signal_list]
    elif isinstance(audio_signal_list, list):
        if not all(isinstance(s, AudioSignal) for s in audio_signal_list):
            raise ValueError('All input objects must be AudioSignal objects!')
        if not all(s.has_data for s in audio_signal_list):
            raise ValueError('All AudioSignal objects in input list must have data!')
    else:
        raise ValueError('All input objects must be AudioSignal objects!')

    return audio_signal_list


def verify_audio_signal_list_strict(audio_signal_list):
    """
    Verifies that an input (:param:`audio_signal_list`) is a list of :ref:`AudioSignal` objects and
    that they all have the same sample rate and same number of channels. If not true,
    attempts to correct the list (if possible) and returns the corrected list.

    Args:
        audio_signal_list (list): List of :ref:`AudioSignal` objects

    Returns:
        audio_signal_list (list): Verified list of :ref:`AudioSignal` objects, that all have
        the same sample rate and number of channels.

    """
    audio_signal_list = verify_audio_signal_list_lax(audio_signal_list)

    if not all(audio_signal_list[0].sample_rate == s.sample_rate for s in audio_signal_list):
        raise ValueError('All input AudioSignal objects must have the same sample rate!')

    if not all(audio_signal_list[0].num_channels == s.num_channels for s in audio_signal_list):
        raise ValueError('All input AudioSignal objects must have the same number of channels!')

    if not all(audio_signal_list[0].signal_length == s.signal_length for s in audio_signal_list):
        raise ValueError('All input AudioSignal objects must have the same signal length!')

    return audio_signal_list


def verify_separation_base_list(separation_list):
    """
    Verifies that all items in :param:`separation_list` are :ref:`SeparationBase` -derived objects.
    If not so, attempts to correct the list if possible and returns the corrected list.

    Args:
        separation_list (list): List of :ref:`SeparationBase` -derived objects

    Returns:
        separation_list (list): Verified list of :ref:`SeparationBase` -derived objects

    """
    # Lazy load to prevent a circular reference upon initialization
    from ..separation import SeparationBase

    if isinstance(separation_list, SeparationBase):
        separation_list = [separation_list]
    elif isinstance(separation_list, list):
        if not all(issubclass(s, SeparationBase) for s in separation_list):
            raise ValueError('All separation objects must be SeparationBase-derived objects!')
    else:
        raise ValueError('All separation objects must be SeparationBase-derived objects!')

    return separation_list


def verify_mask_separation_base_list(mask_separation_list):
    """
    Verifies that all items in :param:`separation_list` are :ref:`MaskSeparationBase` -derived
    objects. If not so, attempts to correct the list if possible and returns the corrected list.

    Args:
        mask_separation_list (list): List of :ref:`MaskSeparationBase` -derived objects

    Returns:
        separation_list (list): Verified list of :ref:`MaskSeparationBase` -derived objects

    """
    # Lazy load to prevent a circular reference upon initialization
    from ..separation import MaskSeparationBase

    if isinstance(mask_separation_list, MaskSeparationBase):
        mask_separation_list = [mask_separation_list]
    elif isinstance(mask_separation_list, list):
        if not all(issubclass(s, MaskSeparationBase) for s in mask_separation_list):
            raise ValueError('All separation objects must be '
                             'MaskSeparationBase-derived objects! {}'.format(mask_separation_list))
    else:
        raise ValueError('All separation objects must be MaskSeparationBase-derived objects!')

    return mask_separation_list


def _verify_audio_data(audio_data):
    """
    A helper method to make sure that input audio data is formatted correctly. This checks if
    :param:`audio_data` is a numpy array, then if it's all finite
    Args:
        audio_data (:obj:`np.ndarray`): A numpy array with audio-data. Can be  1 or 2 dimensional

    Returns:
        Correctly formatted :param:`audio_data` array or `None` if :param:`audio_data` is `None`

    """
    if audio_data is None:
        return None

    if isinstance(audio_data, list):
        audio_data = np.array(audio_data)

    if not isinstance(audio_data, np.ndarray):
        raise ValueError('Type of audio_data must be of type np.ndarray!')

    if not np.isfinite(audio_data).all():
        raise ValueError('Not all values of audio_data are finite!')

    if audio_data.ndim > 1 \
            and audio_data.shape[constants.CHAN_INDEX] > audio_data.shape[constants.LEN_INDEX]:
        warnings.warn('audio_data is not as we expect it. Transposing signal...')
        audio_data = audio_data.T

    if audio_data.ndim > 2:
        raise ValueError('audio_data cannot have more than 2 dimensions!')

    if audio_data.ndim < 2:
        audio_data = np.expand_dims(audio_data, axis=constants.CHAN_INDEX)

    return audio_data


def _verify_transformation_data(transformation_data):
    if transformation_data is None:
        return None

    if isinstance(transformation_data, list):
        transformation_data = np.array(transformation_data)

    if not isinstance(transformation_data, np.ndarray):
        raise ValueError('Type of transformation_data must be of type np.ndarray!')

    if transformation_data.ndim == 1:
        raise ValueError('Cannot support arrays with less than 2 dimensions!')

    if transformation_data.ndim == 2:
        transformation_data = np.expand_dims(transformation_data, axis=constants.TF_CHAN_INDEX)

    if transformation_data.ndim > 3:
        raise ValueError('Cannot support arrays with more than 3 dimensions!')

    return transformation_data


def print_all_separation_algorithms():
    """
    Displays the list of all separation algorithms in nussl to the console.
    """
    from ..separation import all_separation_algorithms
    print('\n'.join([a.__name__ for a in all_separation_algorithms]))
