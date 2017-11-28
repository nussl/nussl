#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Provides utilities for running nussl algorithms that do not belong to any specific algorithm or that are shared
between algorithms.

"""
from __future__ import division
import numpy as np
import warnings
import base64
import json
import os
import sys
import json

from six.moves.urllib_parse import urljoin
from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlopen, Request

from audio_signal import AudioSignal
from separation import SeparationBase
from separation import MaskSeparationBase
import constants

__all__ = ['find_peak_indices', 'find_peak_values',
           'json_ready_numpy_array', 'json_serialize_numpy_array', 'load_numpy_json', 'json_numpy_obj_hook',
           'add_mismatched_arrays', 'add_mismatched_arrays2D', 'complex_randn',
           '_get_axis', '_verify_audio_signal_list_lax', '_verify_audio_signal_list_strict',
           '_verify_separation_list', '_verify_mask_separation_list']


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
            lower0, upper0 = _set_array_zero_indices(cur_peak_idx[0], zero_dist0, input_array.shape[0])
            lower1, upper1 = _set_array_zero_indices(cur_peak_idx[1], zero_dist1, input_array.shape[1])
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
        return [input_array[i] for i in find_peak_indices(input_array, n_peaks, min_dist, do_min, threshold)]
    else:
        return [input_array[i, j] for i, j in find_peak_indices(input_array, n_peaks, min_dist, do_min, threshold)]


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


def _verify_audio_signal_list_lax(audio_signal_list):
    """
    Verifies that an input (audio_signal_list) is a list of :ref:`AudioSignal` objects. If not so, attempts 
    to correct the list (if possible) and returns the corrected list.
    Args:
        audio_signal_list (list): List of :ref:`AudioSignal` objects

    Returns:
        audio_signal_list (list): Verified list of :ref:`AudioSignal` objects.

    """
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


def _verify_audio_signal_list_strict(audio_signal_list):
    """
    Verifies that an input (audio_signal_list) is a list of :ref:`AudioSignal` objects and that they all have the same
    sample rate and same number of channels. If not true, attempts to correct the list (if possible) and returns 
    the corrected list.
    
    Args:
        audio_signal_list (list): List of :ref:`AudioSignal` objects

    Returns: 
        audio_signal_list (list): Verified list of :ref:`AudioSignal` objects, that all have 
        the same sample rate and number of channels.

    """
    audio_signal_list = _verify_audio_signal_list_lax(audio_signal_list)

    if not all(audio_signal_list[0].sample_rate == s.sample_rate for s in audio_signal_list):
        raise ValueError('All input AudioSignal objects must have the same sample rate!')

    if not all(audio_signal_list[0].num_channels == s.num_channels for s in audio_signal_list):
        raise ValueError('All input AudioSignal objects must have the same number of channels!')

    return audio_signal_list


def _verify_separation_list(separation_list):
    """
    Verifies that all items in `separation_list` are :ref:`SeparationBase` -derived objects. If not so, attempts 
    to correct the list if possible and returns the corrected list.
    
    Args:
        separation_list: (list) List of :ref:`SeparationBase` -derived objects

    Returns:
        separation_list: (list) Verified list of :ref:`SeparationBase` -derived objects

    """
    if isinstance(separation_list, SeparationBase):
        separation_list = [separation_list]
    elif isinstance(separation_list, list):
        if not all(isinstance(s, SeparationBase) for s in separation_list):
            raise ValueError('All separation objects must be SeparationBase-derived objects!')
    else:
        raise ValueError('All separation objects must be SeparationBase-derived objects!')

    return separation_list


def _verify_mask_separation_list(mask_separation_list):
    """
    Verifies that all items in `separation_list` are :ref:`MaskSeparationBase` -derived objects. If not so, attempts 
    to correct the list if possible and returns the corrected list.

    Args:
        mask_separation_list: (list) List of :ref:`MaskSeparationBase` -derived objects

    Returns:
        separation_list: (list) Verified list of :ref:`MaskSeparationBase` -derived objects

    """
    if isinstance(mask_separation_list, MaskSeparationBase):
        mask_separation_list = [mask_separation_list]
    elif isinstance(mask_separation_list, list):
        if not all(isinstance(s, MaskSeparationBase) for s in mask_separation_list):
            raise ValueError('All separation objects must be MaskSeparationBase-derived objects!')
    else:
        raise ValueError('All separation objects must be MaskSeparationBase-derived objects!')

    return mask_separation_list


def download_audio_example(example_name, local_folder):
    """

    Args:
        example_name:
        local_folder:

    Returns:

    """
    file_url = urljoin(constants.NUSSL_EXTRA_AUDIO_URL, example_name)
    _download_file(example_name, file_url, local_folder, constants.NUSSL_EXTRA_AUDIO_URL)


def print_available_audio_files():
    """

    Returns:

    """
    try:
        request = Request(constants.NUSSL_EXTRA_AUDIO_METADATA_URL)

        # Make sure to get the newest data
        request.add_header('Pragma', 'no-cache')
        request.add_header('Cache-Control', 'max-age=0')
        response = urlopen(request)
        data = json.loads(response.read())
        file_metadata = data['nussl Audio File metadata']

        print('{:40} {:15} {:10} {:50}'.format('File Name', 'Duration (sec)', 'Size', 'Description'))
        for f in file_metadata:
            print('{:40} {:<15.1f} {:10} {:50}'.format(f['file_name'], f['file_length_seconds'],
                                                       f['file_size'], f['file_description']))
        print('\nLast updated {}'.format(data['last_updated']))
        print('To download one of these files insert the file name '
              'as the first parameter to nussl.download_audio_example, like so: \n'
              ' >>> nussl.download_audio_example(\'K0140.wav\')')

    except:
        raise URLError('Cannot fetch metadata from {}!'.format(constants.NUSSL_EXTRA_AUDIO_METADATA_URL))


def download_trained_model(model_name, local_folder):
    """

    Args:
        model_name:
        local_folder:

    Returns:

    """
    file_url = urljoin(constants.NUSSL_EXTRA_MODELS_URL, model_name)
    _download_file(model_name, file_url, local_folder, constants.NUSSL_EXTRA_MODELS_URL)


def download_benchmark_file(benchmark_name, local_folder):
    """

    Args:
        benchmark_name:
        local_folder:

    Returns:

    """
    file_url = urljoin(constants.NUSSL_EXTRA_BENCHMARKS_URL, benchmark_name)
    _download_file(benchmark_name, file_url, local_folder, constants.NUSSL_EXTRA_BENCHMARKS_URL)


def _download_file(file_name, url, local_folder, cache_subdir, file_hash=None, cache_dir=None):
    """

    Heavily inspired by and lovingly adapted from keras' `get_file` function:
    https://github.com/fchollet/keras/blob/afbd5d34a3bdbb0916d558f96af197af1e92ce70/keras/utils/data_utils.py#L109

    Returns:

    """
    if cache_dir is None:
        cache_dir = os.path.expanduser(os.path.join('~', '.nussl'))
    datadir_base = os.path.expanduser(cache_dir)

    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.nussl')

    datadir = os.path.join(datadir_base, cache_subdir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    file_path = os.path.join(datadir, file_name)
    print('Saving file at {}'.format(file_path))

    download = False
    if os.path.exists(file_path):
        if file_hash is not None:
            pass

    else:
        download = True

    if download:
        print('Downloading {} from {}'.format(file_name, url))

        def dl_progress(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)

            if percent < 100:
                sys.stdout.write('\r{}...{}%'.format(file_name, percent))
                sys.stdout.flush()

        error_msg = 'URL fetch failure on {}: {} -- {}'

        try:
            try:
                urlretrieve(url, file_path, dl_progress)
            except URLError as e:
                raise Exception(error_msg.format(url, e.errno, e.reason))
            except HTTPError as e:
                raise Exception(error_msg.format(url, e.code, e.msg))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e

    # try:
    #     with open(file_path, 'w') as f:
    #         remote_file = urllib.urlopen(url)
    #         f.write(remote_file.read())
    #         remote_file.close()
    # except Exception as e:
    #     raise e


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
