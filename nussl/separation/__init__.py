"""
Separation algorithms
=====================

Base classes
------------

These classes are used to build every type of source separation
algorithm currently in nussl. They provide helpful utilities
and make it such that the end-user only has to implement
one or two functions to create a new separation algorithm,
depending on what sort of algorithm they are trying to 
implement.

.. automodule:: nussl.separation.base
    :members:
    :autosummary:
    :undoc-members:

Benchmark methods
-----------------

These methods are used for obtaining upper and lower baselines
for source separation algorithms.

.. automodule:: nussl.separation.benchmark
    :members:
    :autosummary:
    :undoc-members:

Deep methods
------------

Deep networks can be used for source separation via these
classes.

.. automodule:: nussl.separation.deep
    :members:
    :autosummary:
    :undoc-members:

Composite methods
-----------------

These are methods that use the output of 
multiple separation algorithms to build better
more robust separation estimates.

.. automodule:: nussl.separation.composite
    :members:
    :autosummary:
    :undoc-members:

Factorization-based methods
---------------------------

The methods use some sort of factorization-based algorithm
like robust principle component analysis, or independent
component analysis to separate the auditory scene.

.. automodule:: nussl.separation.factorization
    :members:
    :autosummary:
    :undoc-members:

Primitive methods
-----------------

These methods are based on primitives - hard-wired perceptual
grouping cues that are used automatically by the brain. 
Primitives were coined by Albert Bregman in the book 
*Auditory Scene Analysis*. 

.. automodule:: nussl.separation.primitive
    :members:
    :autosummary:
    :undoc-members:

Spatial methods
---------------

These methods are based on primitive spatial cues. 

.. automodule:: nussl.separation.spatial
    :members:
    :autosummary:
    :undoc-members:

"""

from .base import (
    SeparationBase, 
    MaskSeparationBase,
    ClusteringSeparationBase,
    SeparationException,
    NMFMixin,
    DeepMixin
)

from . import (
    deep, 
    spatial, 
    benchmark,
    primitive,
    factorization,
    composite
)
