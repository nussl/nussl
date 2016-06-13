"""
Test for nussl/utils.py
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

