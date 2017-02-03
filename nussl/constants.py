#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A repository containing all of the constants frequently used in
this source separation stuff.
"""
import scipy.signal

DEFAULT_SAMPLE_RATE = 44100  #: (int): Default sample rate. 44.1 kHz, CD-quality
DEFAULT_WIN_LEN_PARAM = 0.04  #: (float): Default window length. 40ms
DEFAULT_BIT_DEPTH = 16  #: (int): Default bit depth. 16-bits, CD-quality
DEFAULT_MAX_VAL = 2 ** 16  #: (int): Max value of 16-bit audio file (unsigned)
EPSILON = 1e-16  #: (float): epsilon for determining small values
MAX_FREQUENCY = DEFAULT_SAMPLE_RATE // 2  #: (int): Maximum frequency representable. 22050 Hz

WINDOW_HAMMING = scipy.signal.hamming.__name__  #: (str): Name for calling Hamming window. 'hamming'
WINDOW_RECTANGULAR = 'rectangular'  #: (str): Name for calling Rectangular window. 'rectangular'
WINDOW_HANN = scipy.signal.hann.__name__  #: (str): Name for calling Hann window. 'hann'
WINDOW_BLACKMAN = scipy.signal.blackman.__name__  #: (str): Name for calling Blackman window. 'blackman'

WINDOW_DEFAULT = WINDOW_HAMMING  #: (str): Default window, Hamming.
ALL_WINDOWS = [WINDOW_HAMMING, WINDOW_RECTANGULAR, WINDOW_HANN, WINDOW_BLACKMAN]
"""list(str): list of all available windows in *nussl*
"""

# TODO: put this in a config file
USE_LIBROSA_STFT = True  #: (bool): Whether *nussl* will use librosa's stft function by default
NUMPY_JSON_KEY = "py/numpy.ndarray"  #: (str): key used when turning numpy arrays into json
