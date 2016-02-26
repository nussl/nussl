#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .spectral_utils import *
import Constants
from .audio_signal import AudioSignal
from Duet import Duet
from Nmf import Nmf, DistanceType
from Repet import Repet, RepetType


__version__ = '0.1.4'

__title__ = 'nussl'
__description__ = 'A flexible sound source separation library.'
__uri__ = 'https://github.com/interactiveaudiolab/nussl'

__author__ = 'C. Grief, E. Manilow, F. Pishdadian'
__email__ = 'ethanmanilow2015@u.northwestern.edu'

__license__ = 'MIT'
__copyright__ = 'Copyright (c) 2016 Interactive Audio Lab'
