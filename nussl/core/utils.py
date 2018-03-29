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
import os
import sys
import hashlib
import re
import collections

from six.moves.urllib_parse import urljoin
from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlopen, Request
import numpy as np
import musdb

# commented these out because they were causing circular import errors
# from audio_signal import AudioSignal
# from separation import SeparationBase
# from separation import MaskSeparationBase
import constants

__all__ = ['find_peak_indices', 'find_peak_values',
           'json_ready_numpy_array', 'json_serialize_numpy_array', 'load_numpy_json',
           'json_numpy_obj_hook',
           'add_mismatched_arrays', 'add_mismatched_arrays2D', 'complex_randn',
           '_get_axis',
           'verify_audio_signal_list_lax', 'verify_audio_signal_list_strict',
           'verify_separation_base_list', 'verify_mask_separation_base_list',
           '_verify_audio_data', '_verify_transformation_data']


def find_peak_indices(input_array, n_peaks, min_dist=None, do_min=False, threshold=0.5):
    """
    This function will find the indices of the peaks of an input n-dimensional numpy array.
    This can be configured to find max or min peak indices, distance between the peaks, and
    a lower bound, at which the algorithm will stop searching for peaks (or upper bound if
    searching for max). Use exactly the same as find_peak_values().

    This function currently only accepts 1-D and 2-D numpy arrays.

    Notes:
        * This function only returns the indices of peaks. If you want to find peak values,
        use find_peak_values().

        * min_dist can be an int or a tuple of length 2.
            If input_array is 1-D, min_dist must be an integer.
            If input_array is 2-D, min_dist can be an integer, in which case the minimum
            distance in both dimensions will be equal. min_dist can also be a tuple if
            you want each dimension to have a different minimum distance between peaks.
            In that case, the 0th value in the tuple represents the first dimension, and
            the 1st value represents the second dimension in the numpy array.


    See Also:
        :: find_peak_values() ::

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
    Finds the values of the peaks in a 1-D or 2-D numpy array. Use exactly the same as
    find_peak_indices(). This function will find the values of the peaks of an input
    n-dimensional numpy array.

    This can be configured to find max or min peak values, distance between the peaks, and
    a lower bound, at which the algorithm will stop searching for peaks (or upper bound if
    searching for max).

    This function currently only accepts 1-D and 2-D numpy arrays.

    Notes:
        * This function only returns the indices of peaks. If you want to find peak values,
        use find_peak_values().

        * min_dist can be an int or a tuple of length 2.
            If input_array is 1-D, min_dist must be an integer.
            If input_array is 2-D, min_dist can be an integer, in which case the minimum
            distance in both dimensions will be equal. min_dist can also be a tuple if
            you want each dimension to have a different minimum distance between peaks.
            In that case, the 0th value in the tuple represents the first dimension, and
            the 1st value represents the second dimension in the numpy array.


    See Also:
        :: find_peak_indices() ::

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
    Will add two 1D numpy arrays of different length. If truncate is false, it will expand
    the resultant array to the larger of the two, if True it will truncate the resultant
    array to the smaller of the two.

    Args:
        array1: (np.array) 1D numeric array
        array2: (np.array) 1D numeric array
        truncate: (Bool) If True, will truncate the resultant array to the smaller of the two

    Returns:
        One 1D array added from the two input arrays

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
    Will add two 2D numpy arrays of different length. If truncate is false, it will expand
    the resultant array to the larger of the two, if True it will truncate the resultant
    array to the smaller of the two.

    Args:
        array1: (np.array) 2D numeric array
        array2: (np.array) 2D numeric array
        truncate: (Bool) If True, will truncate the resultant array to the smaller of the two

    Returns:
        One 2D array added from the two input arrays

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
    Returns a complex-valued numpy array of random values with shape `shape`
    Args:
        shape: (tuple) tuple of ints that will be the shape of the resultant complex numpy array

    Returns: (:obj:`np.ndarray`): a complex-valued numpy array of random values with shape `shape`
    """
    return np.random.randn(*shape) + 1j * np.random.randn(*shape)


def _get_axis(array, axis_num, i):
    """
    Will get index 'i' along axis 'axis_num' of a 2- or 3-dimensional numpy array.
    If array has 4+ dimensions or 'axis_num' is larger than number of axes, will return None.
    Args:
        array: 
        axis_num: 
        i: 

    Returns:

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
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def _format(string):
    """ Formats a class name correctly for checking function and class names.
        Strips all non-alphanumeric chars and makes lowercase.
    """
    return str(filter(str.isalnum, string)).lower()


def audio_signals_to_mudb_track(mixture, sources_dict, targets_dict):
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
    Verifies that an input (audio_signal_list) is a list of :ref:`AudioSignal` objects.
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
    Verifies that an input (audio_signal_list) is a list of :ref:`AudioSignal` objects and
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
    Verifies that all items in `separation_list` are :ref:`SeparationBase` -derived objects.
    If not so, attempts to correct the list if possible and returns the corrected list.

    Args:
        separation_list: (list) List of :ref:`SeparationBase` -derived objects

    Returns:
        separation_list: (list) Verified list of :ref:`SeparationBase` -derived objects

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
    Verifies that all items in `separation_list` are :ref:`MaskSeparationBase` -derived objects.
    If not so, attempts to correct the list if possible and returns the corrected list.

    Args:
        mask_separation_list: (list) List of :ref:`MaskSeparationBase` -derived objects

    Returns:
        separation_list: (list) Verified list of :ref:`MaskSeparationBase` -derived objects

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
    `audio_data` is a numpy array, then if it's all finite
    Args:
        audio_data (:obj:`np.ndarray`): A numpy array with audio-data. Can be  1 or 2 dimensional

    Returns:
        Correctly formatted `audio_data` array or `None` if `audio_data` is `None`

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


def print_available_audio_files():
    """gets a list of available audio files for download from the server and displays them
    to the user.

    Args:

    Returns:

    Example:
        nussl.utils.print_available_audio_files()

        File Name                                Duration (sec)  Size       Description
        dev1_female3_inst_mix.wav                10.0            1.7MiB     Instantaneous mixture of three female speakers talking in a stereo field.
        dev1_female3_synthconv_130ms_5cm_mix.wav 10.0            1.7MiB     Three female speakers talking in a stereo field, with 130ms of inter-channel delay.
        K0140.wav                                5.0             431.0KiB   Acoustic piano playing middle C.
        K0149.wav                                5.0             430.0KiB   Acoustic piano playing the A above middle C. (A440)

        Last updated 2018-03-06

    """
    try:
        data = _download_metadata_file(constants.NUSSL_EXTRA_AUDIO_METADATA_URL)
        file_metadata = data['nussl Audio File metadata']

        print('{:40} {:15} {:10} {:50}'.format('File Name', 'Duration (sec)',
                                               'Size', 'Description'))
        for f in file_metadata:
            print('{:40} {:<15.1f} {:10} {:50}'.format(f['file_name'], f['file_length_seconds'],
                                                       f['file_size'], f['file_description']))
        print('\nLast updated {}'.format(data['last_updated']))
        print('To download one of these files insert the file name '
              'as the first parameter to nussl.download_audio_example, like so: \n'
              ' >>> nussl.download_audio_example(\'K0140.wav\')')

    except:
        raise URLError('Cannot fetch metadata from {}!'
                       .format(constants.NUSSL_EXTRA_AUDIO_METADATA_URL))


def print_available_trained_models():
    """gets a list of available audio files for download from the server and displays them
    to the user.

    Args:

    Returns:

    Example:
        nussl.utils.print_available_trained_models()

        File Name                                For Class            Size       Description
        deep_clustering_model.h5                 DeepClustering       48.1MiB    example Deep Clustering Keras model
        deep_clustering_vocal_44k_long.model     DeepClustering       90.2MiB    trained DC model for vocal extraction

        Last updated 2018-03-06

    """
    try:
        data = _download_metadata_file(constants.NUSSL_EXTRA_MODEL_METADATA_URL)
        file_metadata = data['nussl Models metadata']

        print('{:40} {:20} {:10} {:50}'.format('File Name', 'For Class', 'Size', 'Description'))
        for f in file_metadata:
            print('{:40} {:20} {:10} {:50}'.format(f['file_name'], f['for_class'],
                                                   f['file_size'], f['file_description']))
        print('\nLast updated {}'.format(data['last_updated']))
        print('To download one of these files insert the file name '
              'as the first parameter to nussl.download_trained_model, like so: \n'
              ' >>> nussl.download_trained_model(\'deep_clustering_model.h5\')')

    except Exception:
        raise URLError('Cannot fetch metadata from {}!'
                       .format(constants.NUSSL_EXTRA_MODEL_METADATA_URL))


def print_available_benchmark_files():
    """gets a list of available audio files for download from the server and displays them
    to the user.

    Args:

    Returns:

    Example:
        nussl.utils.print_available_benchmark_files()

        File Name                                For Class            Size       Description
        benchmark.npy                            example              11.0B      example benchmark file
        example.npy                              test                 13.0B      test example

        Last updated 2018-03-06

    """
    try:
        data = _download_metadata_file(constants.NUSSL_EXTRA_BENCHMARK_METADATA_URL)
        file_metadata = data['nussl Benchmarks metadata']

        print('{:40} {:20} {:10} {:50}'.format('File Name', 'For Class', 'Size', 'Description'))
        for f in file_metadata:
            print('{:40} {:20} {:10} {:50}'.format(f['file_name'], f['for_class'],
                                                   f['file_size'], f['file_description']))
        print('\nLast updated {}'.format(data['last_updated']))
        print('To download one of these files insert the file name '
              'as the first parameter to nussl.download_benchmark_file, like so: \n'
              ' >>> nussl.download_benchmark_file(\'example.npy\')')

    except Exception:
        raise URLError('Cannot fetch metadata from {}!'
                       .format(constants.NUSSL_EXTRA_BENCHMARK_METADATA_URL))


def _download_metadata_file(url):
    request = Request(url)

    # Make sure to get the newest data
    request.add_header('Pragma', 'no-cache')
    request.add_header('Cache-Control', 'max-age=0')
    response = urlopen(request)
    return json.loads(response.read())


def download_audio_example(example_name, local_folder=None):
    """downloads the specified Audio file from the `nussl` server.

    Args:
        example_name: (String) name of the audio file to download
        local_folder: (String) path to local folder in which to download the file,
            defaults to regular `nussl` directory

    Returns:
        (String) path to the downloaded file

    Example:
        input_file = nussl.utils.download_audio_example('K0140.wav')
        music = AudioSignal(input_file, offset=45, duration=20)

    """
    file_metadata = _download_metadata(example_name, 'audio')

    file_hash = file_metadata['file_hash']

    file_url = urljoin(constants.NUSSL_EXTRA_AUDIO_URL, example_name)
    result = _download_file(example_name, file_url, local_folder, 'audio', file_hash=file_hash)

    return result


def download_trained_model(model_name, local_folder=None):
    """downloads the specified Model file from the NUSSL server.

    Args:
        model_name: (String) name of the trained model to download
        local_folder: (String)  a local folder in which to download the file,
        defaults to regular NUSSL directory

    Returns:
        (String) path to the downloaded file

    Example:
        model_path = nussl.utils.download_trained_model('deep_clustering_vocal_44k_long.model')

    """
    file_metadata = _download_metadata(model_name, 'model')

    file_hash = file_metadata['file_hash']

    file_url = urljoin(constants.NUSSL_EXTRA_MODELS_URL, model_name)
    result = _download_file(model_name, file_url, local_folder, 'models', file_hash=file_hash)

    return result


def download_benchmark_file(benchmark_name, local_folder=None):
    """downloads the specified Benchmark file from the NUSSL server.

    Args:
        benchmark_name: (String) name of the benchmark to download
        local_folder: (String)  a local folder in which to download the file,
        defaults to regular NUSSL directory

    Returns:
        (String) path to the downloaded file

    """
    file_metadata = _download_metadata(benchmark_name, 'benchmark')

    file_hash = file_metadata['file_hash']

    file_url = urljoin(constants.NUSSL_EXTRA_BENCHMARKS_URL, benchmark_name)
    result = _download_file(benchmark_name, file_url, local_folder, 'benchmarks',
                            file_hash=file_hash)

    return result


def _download_metadata(file_name, file_type):
    """downloads the metadata file for the specified file type and finds the entry for
    the specified file.

    Args:
        file_name: (String) name of the file's metadata to locate
        file_type: (String) type of file, determines the JSON file to download.
        One of [audio, model, benchmark].

    Returns:
        (dict) metadata for the specified file, or None if it could not be located.


    """

    metadata_urls = {
        'audio': constants.NUSSL_EXTRA_AUDIO_METADATA_URL,
        'benchmark': constants.NUSSL_EXTRA_BENCHMARK_METADATA_URL,
        'model': constants.NUSSL_EXTRA_MODEL_METADATA_URL,
    }

    metadata_labels = {
        'audio': 'nussl Audio File metadata',
        'benchmark': 'nussl Benchmarks metadata',
        'model': 'nussl Models metadata',
    }

    if metadata_urls[file_type]:
        metadata_url = metadata_urls[file_type]
    else:
        # wrong file type, return
        print("metadata: wrong file type.")
        return None

    try:
        request = Request(metadata_url)

        request.add_header('Pragma', 'no-cache')
        request.add_header('Cache-Control', 'max-age=0')
        response = urlopen(request)

        data = json.loads(response.read())
        metadata = data[metadata_labels[file_type]]

        for file_metadata in metadata:
            if file_metadata['file_name'] == file_name:
                return file_metadata

        # TODO: do we raise an exception here if the file isn't found on the server?
        print("Metadata: " + file_type + " file metadata not found on server")
        return None

    except Exception as e:
        raise URLError('Cannot fetch metadata from {}!'
                       .format(constants.NUSSL_EXTRA_AUDIO_METADATA_URL))


def _download_file(file_name, url, local_folder, cache_subdir, file_hash=None, cache_dir=None):
    """

    Heavily inspired by and lovingly adapted from keras' `get_file` function:
    https://github.com/fchollet/keras/blob/afbd5d34a3bdbb0916d558f96af197af1e92ce70/keras/utils/data_utils.py#L109

    Args:
        file_name: (String) name of the file located on the server
        url: (String) url of the file
        local_folder: (String) alternate folder in which to download the file
        cache_subdir: (String) subdirectory of folder in which to download flie
        file_hash: (String) expected hash of downloaded file
        cache_dir:

    Returns:
        (String) local path to downloaded file

    """
    if local_folder not in [None, '']:
        # local folder provided, let's create it if it doesn't exist and use it as datadir
        if not os.path.exists(os.path.expanduser(local_folder)):
            os.makedirs(os.path.expanduser(local_folder))
        datadir = os.path.expanduser(local_folder)

    else:
        if cache_dir is None:
            cache_dir = os.path.expanduser(os.path.join('~', '.nussl'))
        datadir_base = os.path.expanduser(cache_dir)

        if not os.access(datadir_base, os.W_OK):
            datadir_base = os.path.join('/tmp', '.nussl')

        datadir = os.path.join(datadir_base, cache_subdir)
        if not os.path.exists(datadir):
            os.makedirs(datadir)

    file_path = os.path.join(datadir, file_name)

    download = False
    if os.path.exists(file_path):
        if file_hash is not None:
            # compare the provided hash with the hash of the file currently at file_path
            current_hash = _hash_file(file_path)
            # if the hashes are equal, we alreay have the file we need, so don't download
            if file_hash != current_hash:
                print("checked hashes, they're not equal")
                download = True

        else:
            download = True

    else:
        download = True

    if download:
        print('Saving file at {}'.format(file_path))
        print('Downloading {} from {}'.format(file_name, url))

        def _dl_progress(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)

            if percent <= 100:
                sys.stdout.write('\r{}...{}%'.format(file_name, percent))
                sys.stdout.flush()

        error_msg = 'URL fetch failure on {}: {} -- {}'

        try:
            try:
                urlretrieve(url, file_path, _dl_progress)
            except HTTPError as e:
                raise FailedDownloadError(error_msg.format(url, e.code, e.msg))
            except URLError as e:
                raise FailedDownloadError(error_msg.format(url, e.errno, e.reason))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e

        # check hash of received file to see if it matches the provided hash
        if file_hash is not None:
            download_hash = _hash_file(file_path)
            if file_hash != download_hash:
                # the downloaded file is not what it should be. Get rid of it.
                os.remove(file_path)
                raise MismatchedHashError("downloaded file has been deleted "
                                          "because of a hash mismatch.")

        return file_path

    else:
        return file_path


def _hash_file(file_path, chunk_size=65535):
    """

    Args:
        file_path: System path to the file to be hashed
        chunk_size: size of chunks

    Returns:
        file_hash: the SHA256 hashed string in hex

    """
    hasher = hashlib.sha256()

    with open(file_path, 'rb') as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
            hasher.update(chunk)

    return hasher.hexdigest()


class FailedDownloadError(Exception):
    """
    Exception class for failed downloads
    """
    pass


class MismatchedHashError(Exception):
    """
    Exception class for when a computed hash function does match a pre-computed hash.
    """
    pass


# This is wholesale from keras
if sys.version_info[0] == 2:
    def urlretrieve(url, filename, reporthook=None, data=None):
        """Replacement for `urlretrive` for Python 2.
        Under Python 2, `urlretrieve` relies on `FancyURLopener` from legacy
        `urllib` module, known to have issues with proxy management.
        # Arguments
            url: url to retrieve.
            filename: where to store the retrieved data locally.
            reporthook: a hook function that will be called once
                on establishment of the network connection and once
                after each block read thereafter.
                The hook will be passed three arguments;
                a count of blocks transferred so far,
                a block size in bytes, and the total size of the file.
            data: `data` argument passed to `urlopen`.
        """
        def chunk_read(response, chunk_size=8192, reporthook=None):
            content_type = response.info().get('Content-Length')
            total_size = -1
            if content_type is not None:
                total_size = int(content_type.strip())
            count = 0
            while 1:
                chunk = response.read(chunk_size)
                count += 1
                if not chunk:
                    reporthook(count, total_size, total_size)
                    break
                if reporthook:
                    reporthook(count, chunk_size, total_size)
                yield chunk

        response = urlopen(url, data)
        with open(filename, 'wb') as fd:
            for chunk in chunk_read(response, reporthook=reporthook):
                fd.write(chunk)
else:
    from six.moves.urllib.request import urlretrieve  # pylint: disable=g-import-not-at-top
