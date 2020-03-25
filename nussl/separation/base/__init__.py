"""
Base classes for all separation algorithms in nussl.
"""

from .separation_base import SeparationBase, SeparationException
from .mask_separation_base import MaskSeparationBase
from .clustering_separation_base import ClusteringSeparationBase
from .deep_mixin import DeepMixin
from .nmf_mixin import NMFMixin
