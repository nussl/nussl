#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .config import *
from .constants import *
from .utils import *
from .spectral_utils import *

from .audio_signal import AudioSignal

from .transformers import *
from .separation import *
from .evaluation import *

__all__ = ['config', 'constants', 'utils', 'spectral_utils',
           'audio_signal', 'transformers', 'separation', 'evaluation']


__version__ = '0.1.5a11'

version = __version__  # aliasing version
short_version = '.'.join(version.split('.')[:-1])

__title__ = 'nussl'
__description__ = 'A flexible sound source separation library.'
__uri__ = 'https://github.com/interactiveaudiolab/nussl'

__author__ = 'E. Manilow, P. Seetharaman, F. Pishdadian'
__email__ = 'ethanmanilow@u.northwestern.edu'

__license__ = 'MIT'
__copyright__ = 'Copyright (c) 2017 Interactive Audio Lab'
