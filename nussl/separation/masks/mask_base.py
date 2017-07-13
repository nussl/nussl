#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base class for Mask objects. Contains many common utilities used for accessing masks. The mask itself is
represented under the hood as a three dimensional numpy ``ndarray`` object. The dimensions are 
``[NUM_FREQ, NUM_HOPS, NUM_CHAN]``. Safe accessors for these array indices are in :ref:`constants` as well as 
below.

"""

import numpy as np

import nussl
import nussl.utils
import nussl.constants


class MaskBase(object):
    """
    Args:
        mask (:obj:`np.ndarray`): A 2- or 3-dimensional numpy ``ndarray`` representing a mask.
    """
    def __init__(self, mask):
        self._mask = None
        self.mask = mask

    @property
    def mask(self):
        """
        The actual mask. This is represented as a three dimensional numpy ``ndarray`` object.
        The input gets validated by ``_validate_mask``. In the case of :ref:`binary_mask` the validation checks that
        the values are all 1 or 0 (or bools), in the case of :ref:`soft_mask` the validation checks that all values
        are within the domain [0.0, 1.0].
        
        This base class will throw a ``NotImplementedError`` if instantiated directly.
        
        Raises:
            ``ValueError`` if ``mask.ndim`` is less than 2 or greater than 3, or if values fail validation.
            ``NotImplementedError`` if instantiated directly.

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
        Gets mask channel ``n`` and returns it as a 2D ``np.ndarray``
        Args:
            n (int): Channel index to return (0-based).

        Returns:
            :obj:`np.array` with the mask channel
            
        Raises:
            ``AttributeError`` if ``self.mask`` is ``None``
            ``ValueError`` if ``n`` is less than 0 or greater than the number of channels that this mask object has.

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
        (int) Number of time hops that this mask represents.

        """
        if self.mask is None:
            raise AttributeError('Cannot get length of BinaryMask when there is no mask data!')
        return self.mask.shape[nussl.constants.STFT_LEN_INDEX]

    @property
    def height(self):
        """
        (int) Number of frequency bins this mask has.

        """
        if self.mask is None:
            raise AttributeError('Cannot get height of BinaryMask when there is no mask data!')
        return self.mask.shape[nussl.constants.STFT_VERT_INDEX]

    @property
    def num_channels(self):
        """
        (int) Number of channels this mask has.

        """
        if self.mask is None:
            raise AttributeError('Cannot get num_channels of BinaryMask when there is no mask data!')
        return self.mask.shape[nussl.constants.STFT_CHAN_INDEX]

    @property
    def shape(self):
        """
        (tuple) Returns the shape of the whole mask. Identical to ``np.ndarray.shape()``.

        """
        if self.mask is None:
            raise AttributeError('Cannot get shape of BinaryMask when there is no mask data!')
        return self.mask.shape

    @property
    def dtype(self):
        """
        (str) Returns the data type of the values of the mask. 

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
