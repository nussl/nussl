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

from . import evaluation
