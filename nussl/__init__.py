#!/usr/bin/env python
# -*- coding: utf-8 -*-
try:
    import torch
    torch_imported = True
except Exception:
    torch_imported = False


try:
    import vamp
    vamp_imported = True
except Exception:
    vamp_imported = False

class ImportErrorClass(object):
    def __init__(self, lib, **kwargs):
        raise ImportError(f'Cannot import {type(self).__name__} because {lib} is not installed')

from .core.constants import *
from .core.audio_signal import AudioSignal
from .core import utils, efz_utils, jupyter_utils
# from .separation import *
# from . import evaluation
# from . import transformers
# from . import deep
# from . import core 
# from . import separation

__version__ = '0.1.6'

version = __version__  # aliasing version
short_version = '.'.join(version.split('.')[:-1])

__title__ = 'nussl'
__description__ = 'A flexible sound source separation library.'
__uri__ = 'https://github.com/interactiveaudiolab/nussl'

__author__ = 'E. Manilow, P. Seetharaman, F. Pishdadian, N. Shelly, B. Pardo'
__email__ = 'ethanmanilow@u.northwestern.edu'

__license__ = 'MIT'
__copyright__ = 'Copyright (c) 2018 Interactive Audio Lab'
