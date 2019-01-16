#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A repository containing all of the constants frequently used in
this wacky, mixed up source separation stuff.
"""
import scipy.signal
from six.moves.urllib_parse import urljoin

__all__ = ['DEFAULT_SAMPLE_RATE', 'DEFAULT_WIN_LEN_PARAM', 'DEFAULT_BIT_DEPTH',
           'DEFAULT_MAX_VAL', 'EPSILON', 'MAX_FREQUENCY',
           'WINDOW_HAMMING', 'WINDOW_RECTANGULAR', 'WINDOW_HANN',
           'WINDOW_BLACKMAN', 'WINDOW_TRIANGULAR', 'WINDOW_DEFAULT',
           'ALL_WINDOWS', 'NUMPY_JSON_KEY', 'LEN_INDEX', 'CHAN_INDEX',
           'STFT_VERT_INDEX', 'STFT_LEN_INDEX', 'STFT_CHAN_INDEX']

DEFAULT_SAMPLE_RATE = 44100  #: (int): Default sample rate. 44.1 kHz, CD-quality
DEFAULT_WIN_LEN_PARAM = 0.04  #: (float): Default window length. 40ms
DEFAULT_WIN_LENGTH = 2048  #: (int): Default window length, 2048 samples.
DEFAULT_BIT_DEPTH = 16  #: (int): Default bit depth. 16-bits, CD-quality
DEFAULT_MAX_VAL = 2 ** 16  #: (int): Max value of 16-bit audio file (unsigned)
EPSILON = 1e-16  #: (float): epsilon for determining small values
MAX_FREQUENCY = DEFAULT_SAMPLE_RATE // 2  #: (int): Maximum frequency representable. 22050 Hz

WINDOW_HAMMING = scipy.signal.hamming.__name__  #: (str): Name for calling Hamming window. 'hamming'
WINDOW_RECTANGULAR = 'rectangular'  #: (str): Name for calling Rectangular window. 'rectangular'
WINDOW_HANN = scipy.signal.hann.__name__  #: (str): Name for calling Hann window. 'hann'
WINDOW_BLACKMAN = scipy.signal.blackman.__name__  #: (str): Name for calling Blackman window. 'blackman'
WINDOW_TRIANGULAR = 'triangular'  #: (str): Name for calling Triangular window. 'triangular'

WINDOW_DEFAULT = WINDOW_HAMMING  #: (str): Default window, Hamming.
ALL_WINDOWS = [WINDOW_HAMMING, WINDOW_RECTANGULAR, WINDOW_HANN, WINDOW_BLACKMAN, WINDOW_TRIANGULAR]
"""list(str): list of all available windows in *nussl*
"""

NUMPY_JSON_KEY = "py/numpy.ndarray"  #: (str): key used when turning numpy arrays into json

BINARY_MASK = 'binary'
""" String alias for setting this object to return :class:`separation.masks.binary_mask.BinaryMask` objects
"""

SOFT_MASK = 'soft'
""" String alias for setting this object to return :class:`separation.masks.soft_mask.SoftMask` objects
"""

# ############# Array Indices ############# #

# audio_data
LEN_INDEX  = 1  #: (int): Index of the number of samples in an audio signal. Used in :ref:`audio_signal`
CHAN_INDEX = 0  #: (int): Index of the number of channels in an audio signal. Used in :ref:`audio_signal`

# stft_data
STFT_VERT_INDEX = 0
"""
(int) Index of the number of frequency (vertical) values in a time-frequency representation. 
Used in :ref:`audio_signal` and in :ref:`mask_base`.
"""
STFT_LEN_INDEX  = 1
"""
(int) Index of the number of time (horizontal) hops in a time-frequency representation. 
Used in :ref:`audio_signal` and in :ref:`mask_base`.
"""
STFT_CHAN_INDEX = 2
"""
(int) Index of the number of channels in a time-frequency representation. 
Used in :ref:`audio_signal` and in :ref:`mask_base`.
"""

# ############# nussl-extras urls ############# #

NUSSL_EFZ_BASE_URL = 'http://nussl.ci.northwestern.edu/'
NUSSL_EFZ_STATIC_BASE_URL = urljoin(NUSSL_EFZ_BASE_URL, 'static/')
NUSSL_EFZ_AUDIO_URL = urljoin(NUSSL_EFZ_STATIC_BASE_URL, 'audio/')
NUSSL_EFZ_MODELS_URL = urljoin(NUSSL_EFZ_STATIC_BASE_URL, 'trained_models/')
NUSSL_EFZ_BENCHMARKS_URL = urljoin(NUSSL_EFZ_STATIC_BASE_URL, 'benchmarks/')
NUSSL_EFZ_AUDIO_METADATA_URL = urljoin(NUSSL_EFZ_BASE_URL, 'audio-metadata.json')
NUSSL_EFZ_BENCHMARK_METADATA_URL = urljoin(NUSSL_EFZ_BASE_URL, 'benchmark-metadata.json')
NUSSL_EFZ_MODEL_METADATA_URL = urljoin(NUSSL_EFZ_BASE_URL, 'model-metadata.json')

USE_LIBROSA_STFT = False  #: (bool): Whether *nussl* will use librosa's stft function by default


# ############# MUSDB interface ############### #
STEM_TARGET_DICT = {'vocals': {'vocals': 1},
                    'drums': {'drums': 1},
                    'bass': {'bass': 1},
                    'other': {'other': 1},
                    'accompaniment': {'bass': 1, 'drums': 1, 'other': 1}}

VOX_ACC_DICT = {'vocals': {'vocals': 1}, 'accompaniment': {'accompaniment': 1}}
