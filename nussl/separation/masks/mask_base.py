#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Abstract class for Mask. 

"""

import numpy as np

import nussl
import nussl.utils
import nussl.constants


class MaskBase(object):
    """
    Base class for Mask objects. Contains many common 
    """
    def __init__(self, mask):
        self._mask = None
        self.mask = mask

    @property
    def mask(self):
        """
        

        Returns:

        """
        return self._mask

    @mask.setter
    def mask(self, value):
        assert isinstance(value, np.ndarray), 'Type of self.mask must be np.ndarray!'

        if value.ndim == 1:
            raise ValueError('Cannot support arrays with less than 2 dimensions!')

        if value.ndim == 2:
            value = np.expand_dims(value, axis=nussl.constants.STFT_CHAN_INDEX)

        if value.ndim > 3:
            raise ValueError('Cannot support arrays with more than 3 dimensions!')

        self._mask = self._validate_mask(value)

    def get_channel(self, n):
        """
        
        Args:
            n: 

        Returns:

        """
        if self.mask is None:
            raise AttributeError('Cannot get channel {} when mask has no data!'.format(n))

        if n >= self.num_channels:
            raise ValueError( 'Cannot get channel {0} when this object only has {1} channels! (0-based)'
                              .format(n, self.num_channels))

        if n < 0:
            raise ValueError('Cannot get channel {}. This will cause unexpected results!'.format(n))

        return nussl.utils._get_axis(self.mask, nussl.constants.STFT_CHAN_INDEX, n)

    @property
    def length(self):
        """

        Returns:

        """
        if self.mask is None:
            raise AttributeError('Cannot get length of BinaryMask when there is no mask data!')
        return self.mask.shape[nussl.constants.STFT_LEN_INDEX]

    @property
    def height(self):
        """

        Returns:

        """
        if self.mask is None:
            raise AttributeError('Cannot get height of BinaryMask when there is no mask data!')
        return self.mask.shape[nussl.constants.STFT_VERT_INDEX]

    @property
    def num_channels(self):
        """

        Returns:

        """
        if self.mask is None:
            raise AttributeError('Cannot get num_channels of BinaryMask when there is no mask data!')
        return self.mask.shape[nussl.constants.STFT_CHAN_INDEX]

    @property
    def shape(self):
        """

        Returns:

        """
        if self.mask is None:
            raise AttributeError('Cannot get shape of BinaryMask when there is no mask data!')
        return self.mask.shape

    @property
    def dtype(self):
        """

        Returns:

        """
        if self.mask is None:
            raise AttributeError('Cannot get num_channels of BinaryMask when there is no mask data!')
        return self.mask.dtype

    @staticmethod
    def _validate_mask(mask_):
        """
        
        Args:
            mask_: 

        Returns:

        """
        raise NotImplementedError('Cannot call base class! Use BinaryMask or SoftMask!')
