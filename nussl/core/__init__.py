"""
Core
====

AudioSignals
------------
.. autoclass:: nussl.core.AudioSignal
    :members:
    :autosummary:

Masks
-----
.. automodule:: nussl.core.masks
    :members:
    :autosummary:

Constants
------------
.. automodule:: nussl.core.constants
    :members:
    :autosummary:

External File Zoo
-----------------
.. automodule:: nussl.core.efz_utils
    :members:
    :autosummary:


General utilities
-----------------
.. automodule:: nussl.core.utils
    :members:
    :autosummary:

Audio effects
-------------
.. automodule:: nussl.core.effects
    :members:
    :autosummary:

Mixing
------
.. automodule:: nussl.core.mixing
    :members:
    :autosummary:

Playing and embedding audio
---------------------------
.. automodule:: nussl.core.play_utils
    :members:
    :autosummary:

Checkpoint migration (backwards compatability)
----------------------------------------------
.. automodule:: nussl.core.migration
    :members:
    :autosummary:

"""

from .audio_signal import AudioSignal, STFTParams
from . import constants
from . import efz_utils
from . import play_utils
from . import utils
from . import mixing
from . import masks

__all__ = [
    'AudioSignal', 
    'STFTParams',
    'constants',
    'efz_utils',
    'play_utils',
    'utils',
    'mixing'
    'masks',
]
