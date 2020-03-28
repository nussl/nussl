"""
Base for all methods
--------------------

.. autoclass:: nussl.separation.SeparationBase
    :members:
    :autosummary:

Base for masking-based methods
------------------------------

.. autoclass:: nussl.separation.MaskSeparationBase
    :members:
    :autosummary:

Base for clustering-based methods
---------------------------------

.. autoclass:: nussl.separation.ClusteringSeparationBase
    :members:
    :autosummary:

Mix-in for NMF-based methods
----------------------------

.. autoclass:: nussl.separation.NMFMixin
    :members:
    :autosummary:

Mix-in for deep methods
-----------------------

.. autoclass:: nussl.separation.DeepMixin
    :members:
    :autosummary:

"""

from .separation_base import SeparationBase, SeparationException
from .mask_separation_base import MaskSeparationBase
from .clustering_separation_base import ClusteringSeparationBase
from .deep_mixin import DeepMixin
from .nmf_mixin import NMFMixin
