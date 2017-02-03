#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A repository containing all of the constants frequently used in
this source separation stuff.
"""
import scipy.signal

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_WIN_LEN_PARAM = 0.04
DEFAULT_BIT_DEPTH = 16
DEFAULT_MAX_VAL = 2 ** 16
EPSILON = 1e-16
MAX_FREQUENCY = DEFAULT_SAMPLE_RATE // 2

WINDOW_HAMMING = scipy.signal.hamming.__name__  # 'hamming'
WINDOW_RECTANGULAR = 'rectangular'
WINDOW_HANN = scipy.signal.hann.__name__  # 'hann'
WINDOW_BLACKMAN = scipy.signal.blackman.__name__  # 'blackman'

WINDOW_DEFAULT = WINDOW_HAMMING

USE_LIBROSA_STFT = True  # TODO: put this in a config file
NUMPY_JSON_KEY = "py/numpy.ndarray"
