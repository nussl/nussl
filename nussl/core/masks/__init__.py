"""
init for masks files
"""

from .mask_base import MaskBase
from .binary_mask import BinaryMask
from .soft_mask import SoftMask

__all__ = ['MaskBase', 'BinaryMask', 'SoftMask']
