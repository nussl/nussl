#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nussl.config import *
from nussl.constants import *
from nussl.utils import *
from nussl.spectral_utils import *
from nussl.audio_signal import AudioSignal
from nussl.separation import *
from nussl.separation.masks import *
from nussl.evaluation import *
from nussl.Nmf import *

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
