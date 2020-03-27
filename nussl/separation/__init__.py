"""
Separation algorithms
=====================

Benchmark methods
-----------------

.. automodule:: nussl.separation.benchmark
    :members:
    :autosummary:

Deep methods
------------

.. automodule:: nussl.separation.deep
    :members:
    :autosummary:

"""

from .base import (
    SeparationBase, 
    MaskSeparationBase,
    ClusteringSeparationBase,
    SeparationException
)

from . import (
    deep, 
    spatial, 
    benchmark,
    primitive,
    factorization,
    composite
)
