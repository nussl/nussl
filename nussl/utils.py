"""
Provides utilities for running nussl algorithms that do not belong to any specific algorithm or that are shared
between algorithms.

"""
import numpy as np
import warnings


def find_peak_indices(input_array, n_peaks, min_dist=None, do_min=False, threshold=0.5):
    """

    Args:
        input_array: a 1- or 2- dimensional array that will be inspected
        n_peaks: (int) maximum number of peaks to find
        min_dist: (int) minimum distance between peaks. Default value: len(input_array) / 4
        do_min: (bool) finds minimum values instead of maximums
        threshold: (float)

    Returns:

    """
    input_array = np.array(input_array, dtype=float)

    if input_array.ndim > 2:
        raise ValueError('Cannot find peak indices on data greater than 2 dimensions!')

    is_1d = input_array.ndim == 1
    zero_dist = zero_dist0 = zero_dist1 = 0

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
            lower, upper = _set_array_zeroe_indices(cur_peak_idx, zero_dist, len(input_array))
            input_array[lower:upper] = 0
        else:
            lower0, upper0 = _set_array_zeroe_indices(cur_peak_idx, zero_dist0, input_array.shape[0])
            lower1, upper1 = _set_array_zeroe_indices(cur_peak_idx, zero_dist1, input_array.shape[1])
            input_array[lower0:upper0, lower1:upper1] = 0

        if np.sum(input_array) == 0.0:
            break

    return peak_indices


def _set_array_zeroe_indices(index, zero_distance, max_len):
    lower = index - zero_distance - 1
    upper = index + zero_distance + 1
    lower = 0 if lower < 0 else lower
    upper = max_len if upper >= max_len else upper
    return lower, upper


def find_peak_values(input_array, n_peaks, min_dist=None, do_min=False, threshold=0.5):
    """

    Args:
        input_array:
        n_peaks:
        min_dist:
        do_min:
        threshold:

    Returns:

    """
    return [input_array[i] for i in find_peak_indices(input_array, n_peaks, min_dist, do_min, threshold)]