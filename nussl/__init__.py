#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .spectral_utils import *
from constants import *
from .audio_signal import AudioSignal
from separation_base import SeparationBase
from Duet import Duet
from Nmf import NMF, DistanceType
from repet import Repet
from repet_sim import RepetSim


__version__ = '0.1.5a4'

__title__ = 'nussl'
__description__ = 'A flexible sound source separation library.'
__uri__ = 'https://github.com/interactiveaudiolab/nussl'

__author__ = 'C. Grief, E. Manilow, F. Pishdadian'
__email__ = 'ethanmanilow2015@u.northwestern.edu'

__license__ = 'MIT'
__copyright__ = 'Copyright (c) 2016 Interactive Audio Lab'
