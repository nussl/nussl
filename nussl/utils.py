"""
Provides utilities for running nussl algorithms that do not belong to any specific algorithm or that are shared
between algorithms.

"""
import numpy as np
import warnings
import base64
import json
import constants


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
        cur_peak_idx = np.argmax(input_array)
        peak_indices.append(cur_peak_idx)

        # zero out peak and its surroundings
        if is_1d:
            lower, upper = _set_array_zero_indices(cur_peak_idx, zero_dist, len(input_array))
            input_array[lower:upper] = 0
        else:
            lower0, upper0 = _set_array_zero_indices(cur_peak_idx, zero_dist0, input_array.shape[0])
            lower1, upper1 = _set_array_zero_indices(cur_peak_idx, zero_dist1, input_array.shape[1])
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
    return [input_array[i] for i in find_peak_indices(input_array, n_peaks, min_dist, do_min, threshold)]


def json_ready_numpy_array(array):
    """
    Adapted from:
    http://stackoverflow.com/a/27948073/5768001
    Args:
        array: np array to make json ready.

    Returns:

    """
    if isinstance(array, np.ndarray):
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
    Will add two 1-D numpy arrays of different length. If truncate is false, it will expand
    the resultant array to the larger of the two, if True it will truncate the resultant
    array to the smaller of the two.

    Args:
        array1: (np.array) 1-D numeric array
        array2: (np.array) 1-D numeric array
        truncate: (Bool) If True, will truncate the resultant array to the smaller of the two

    Returns:
        One array added from the two input arrays

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
