#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

import numpy as np
import warnings

import mask_base
import binary_mask


class SoftMask(mask_base.MaskBase):
    """
    A simple class for making a soft mask. The soft mask is represented as a numpy array of floats between
    0.0 and 1.0, inclusive. 
    """

    def __init__(self, input_mask):
        super(SoftMask, self).__init__(input_mask)

    @staticmethod
    def _validate_mask(mask_):
        assert isinstance(mask_, np.ndarray), 'Mask must be a numpy array!'

        if np.max(mask_) > 1.0 or np.min(mask_) < 0.0:
            # raise ValueError('All values must be between [0.0, 1.0] for SoftMask!')
            # TODO: maybe normalize instead of throwing a warning/error?
            # warnings.warn('All values must be between [0.0, 1.0] for SoftMask! max/min={}/{}'.format(np.max(mask_),
            #                                                                                          np.min(mask_)))
            mask_ /= np.max(mask_)

        if mask_.dtype.kind not in np.typecodes['AllFloat']:
            raise ValueError('Mask must have type: float! Maybe you want BinaryMask?')

        return mask_

    def mask_to_binary(self, threshold=0.5):
        """
        
        Returns:

        """
        return binary_mask.BinaryMask(self.mask > threshold)

    def inverse_mask(self, channel=None):
        """

        Returns:

        """
        if channel is None:
            new_mask = np.abs(1 - self.mask)
            return SoftMask(new_mask)
        else:
            # TODO: this does not give the 2D np.array!
            new_mask = np.abs(1 - self.get_channel(channel))
            return SoftMask(new_mask)
