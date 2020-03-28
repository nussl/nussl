"""
Cluster sources by timbre
-------------------------

.. autoclass:: nussl.separation.primitive.TimbreClustering
    :autosummary:

Foreground/background via 2DFT
------------------------------

.. autoclass:: nussl.separation.primitive.FT2D
    :autosummary:

Harmonic/percussive separation
------------------------------

.. autoclass:: nussl.separation.primitive.HPSS
    :autosummary:

Foreground/background via REPET
-------------------------------

.. autoclass:: nussl.separation.primitive.Repet
    :autosummary:

Foreground/background via REPET-SIM
-----------------------------------

.. autoclass:: nussl.separation.primitive.RepetSim
    :autosummary:

Vocal melody extraction via Melodia
-----------------------------------

.. autoclass:: nussl.separation.primitive.Melodia
    :autosummary:

"""

from .timbre import TimbreClustering
from .ft2d import FT2D
from .hpss import HPSS
from .repet import Repet
from .repet_sim import RepetSim
from .melodia import Melodia
