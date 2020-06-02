"""
Datasets
========

Base class
----------
.. autoclass:: nussl.datasets.BaseDataset
    :members:
    :autosummary:

MUSDB18
-------
.. autoclass:: nussl.datasets.MUSDB18
    :members:
    :autosummary:

WHAM
----
.. autoclass:: nussl.datasets.WHAM
    :members:
    :autosummary:

FUSS
----
.. autoclass:: nussl.datasets.FUSS
    :members:
    :autosummary:

Scaper
------
.. autoclass:: nussl.datasets.Scaper
    :members:
    :autosummary:

MixSourceFolder
---------------
.. autoclass:: nussl.datasets.MixSourceFolder
    :members:
    :autosummary:

OnTheFly
--------
.. autoclass:: nussl.datasets.OnTheFly
    :members:
    :autosummary:

Data transforms
---------------
.. automodule:: nussl.datasets.transforms
    :members:
    :autosummary:

"""

from .base_dataset import BaseDataset
from .hooks import (
    MUSDB18, 
    MixSourceFolder, 
    Scaper, 
    WHAM, 
    FUSS, 
    OnTheFly
)
from . import transforms
