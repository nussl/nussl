"""
Tests for nussl/utils.py
"""

import unittest
import nussl
import numpy as np
from scipy import signal

class TestUtils(unittest.TestCase):
    """

    """

    def test_find_peak_indices(self):
        min_idx = int(np.pi * 100)
        triangle = np.abs(signal.sawtooth(np.arange(0, 10, 0.01)))

    def test_add_mismatched_arrays(self):
        long_array = np.ones((20,))
        short_array = np.arange(10)
        expected_result = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)

        # Test basic cases
        result = nussl.add_mismatched_arrays(long_array, short_array)
        assert all(np.equal(result, expected_result))

        result = nussl.add_mismatched_arrays(short_array, long_array)
        assert all(np.equal(result, expected_result))

        expected_result = expected_result[:len(short_array)]

        result = nussl.add_mismatched_arrays(long_array, short_array, truncate=True)
        assert all(np.equal(result, expected_result))

        result = nussl.add_mismatched_arrays(short_array, long_array, truncate=True)
        assert all(np.equal(result, expected_result))

        # Test complex casting
        short_array = np.arange(10, dtype=complex)
        expected_result = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=complex)

        result = nussl.add_mismatched_arrays(long_array, short_array)
        assert all(np.equal(result, expected_result))

        result = nussl.add_mismatched_arrays(short_array, long_array)
        assert all(np.equal(result, expected_result))

        expected_result = expected_result[:len(short_array)]

        result = nussl.add_mismatched_arrays(long_array, short_array, truncate=True)
        assert all(np.equal(result, expected_result))

        result = nussl.add_mismatched_arrays(short_array, long_array, truncate=True)
        assert all(np.equal(result, expected_result))

        # Test case where arrays are equal length
        short_array = np.ones((15,))
        expected_result = short_array * 2

        result = nussl.add_mismatched_arrays(short_array, short_array)
        assert all(np.equal(result, expected_result))

        result = nussl.add_mismatched_arrays(short_array, short_array, truncate=True)
        assert all(np.equal(result, expected_result))
