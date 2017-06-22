#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Binary Mask class
"""

import numpy as np

import mask_base


class BinaryMask(mask_base.MaskBase):
    """
    Class for creating a Binary Mask of a time-frequency representation of the audio. 
    """

    def __init__(self, mask):
        super(BinaryMask, self).__init__(mask)


    @staticmethod
    def _validate_mask(mask_):
        assert isinstance(mask_, np.ndarray), 'Mask must be a numpy array!'

        if mask_.dtype == np.bool:
            # This is perfect, do nothing here
            return mask_
        elif mask_.dtype.kind in np.typecodes['AllInteger']:
            if np.max(mask_) > 1 or np.min(mask_) < 0:
                raise ValueError('Found values in mask that are not 0 or 1. Mask must be binary!')
        elif mask_.dtype.kind in np.typecodes['AllFloat']:
            tol = 1e-2
            # If we have a float array, ensure that all values are close to 1 or 0
            if not np.all(np.logical_or(np.isclose(mask_, [0], atol=tol), np.isclose(mask_, [1], atol=tol))):
                raise ValueError('All mask values must be close to 0 or 1!')

        return mask_.astype('bool')

    def mask_as_ints(self, channel=None):
        """

        Returns:

        """
        if channel is None:
            return self.mask.astype('int')
        else:
            return self.get_channel(channel).astype('int')

    def inverse_mask(self, channel=None):
        """
        
        Returns:

        """
        if channel is None:
            return BinaryMask(np.logical_not(self.mask))
        else:
            # TODO: this does not give the 2D np.array!
            return BinaryMask(np.logical_not(self.get_channel(channel)))


    @staticmethod
    def mask_to_binary(mask_, threshold):
        """

        Args:
            mask_: 
            threshold: 

        Returns:

        """
        if isinstance(mask_, np.ndarray):
            return mask_ > threshold
        elif isinstance(mask_, mask_base.MaskBase):
            return mask_.mask > threshold
