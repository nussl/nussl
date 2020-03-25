"""
Base class for Mask objects. Contains many common utilities used for accessing masks. The mask itself is
represented under the hood as a three dimensional numpy :obj:`ndarray` object. The dimensions are 
``[NUM_FREQ, NUM_HOPS, NUM_CHAN]``. Safe accessors for these array indices are in :ref:`constants` as well as 
below.

Right now only spectrogram-like masks are supported (note the shape of the :ref:`mask` property), but in future
releases nussl will support masks for representations with different dimensionality requirements.
"""

import numbers

import numpy as np

from .. import utils
from .. import constants


class MaskBase(object):
    """
    Args:
        input_mask (:obj:`np.ndarray`): A 2- or 3-dimensional numpy ``ndarray`` representing a mask.
        
    """
    def __init__(self, input_mask=None, mask_shape=None):
        self._mask = None

        if mask_shape is None and input_mask is None:
            raise ValueError('Cannot initialize mask without mask_shape or input_mask!')
        if mask_shape is not None and input_mask is not None:
            raise ValueError('Cannot initialize mask with both mask_shape and input_mask!')
        if isinstance(input_mask, np.ndarray):
            self.mask = input_mask
        elif isinstance(mask_shape, tuple):
            self.mask = np.zeros(mask_shape)
        else:
            raise ValueError('input_mask must be a np.ndarray, or mask_shape must be a tuple!')

    @property
    def mask(self):
        """        
        The actual mask. This is represented as a three dimensional numpy :obj:`ndarray` object.
        The input gets validated by :func:`_validate_mask`. In the case of 
        :class:`separation.masks.binary_mask.BinaryMask` the validation checks that the values are all 1 or 0 
        (or bools), in the case of :class:`separation.masks.soft_mask.SoftMask` the validation checks that all values
        are within the domain ``[0.0, 1.0]``.
        
        This base class will throw a ``NotImplementedError`` if instantiated directly.
        
        Raises:
            :obj:`ValueError` if :attr:`mask.ndim` is less than 2 or greater than 3, or if values fail validation.
            :obj:`NotImplementedError` if instantiated directly.

        """
        return self._mask

    @mask.setter
    def mask(self, value):
        assert isinstance(value, np.ndarray), 'Type of self.mask must be np.ndarray!'

        if value.ndim == 1:
            raise ValueError('Cannot support arrays with less than 2 dimensions!')

        if value.ndim == 2:
            value = np.expand_dims(value, axis=constants.STFT_CHAN_INDEX)

        if value.ndim > 3:
            raise ValueError('Cannot support arrays with more than 3 dimensions!')

        self._mask = self._validate_mask(value)

    def get_channel(self, ch):
        """
        Gets mask channel ``ch`` and returns it as a 2D :obj:`np.ndarray`
        
        Args:
            ch (int): Channel index to return (0-based).

        Returns:
            :obj:`np.array` with the mask channel
            
        Raises:
            :obj:`ValueError` if ``ch`` is less than 0 or greater than the number of channels that this mask object has.

        """
        if ch >= self.num_channels:
            raise ValueError(
                f'Cannot get channel {ch} for object w/ {self.num_channels} channels!'
                ' (0-based)'
            )

        if ch < 0:
            raise ValueError(f'Cannot get channel {ch}. This will cause unexpected results!')

        return utils._get_axis(self.mask, constants.STFT_CHAN_INDEX, ch)

    @property
    def num_channels(self):
        """
        (int) Number of channels this mask has.

        """
        return self.mask.shape[constants.STFT_CHAN_INDEX]

    @property
    def shape(self):
        """
        (tuple) Returns the shape of the whole mask. Identical to ``np.ndarray.shape()``.

        """
        return self.mask.shape

    @property
    def dtype(self):
        """
        (str) Returns the data type of the values of the mask. 

        """
        return self.mask.dtype

    @staticmethod
    def _validate_mask(mask_):
        """
        Args:
            mask_: 

        Returns:

        """
        raise NotImplementedError('Cannot call base class! Use BinaryMask or SoftMask!')

    @classmethod
    def ones(cls, shape):
        """
        Makes a mask with all ones with the specified shape. Exactly the same as ``np.ones()``.
        Args:
            shape (tuple): Shape of the resultant mask.

        Returns:

        """
        return cls(np.ones(shape))

    @classmethod
    def zeros(cls, shape):
        """
        Makes a mask with all zeros with the specified shape. Exactly the same as ``np.zeros()``.
        Args:
            shape (tuple): Shape of the resultant mask.

        Returns:

        """
        return cls(np.zeros(shape))

    def invert_mask(self):
        """

        Returns:

        """
        raise NotImplementedError('Cannot call base class! Use BinaryMask or SoftMask!')

    def inverse_mask(self):
        """
        Alias for :func:`invert_mask`

        See Also:
            :func:`invert_mask`

        Returns:

        """
        return self.invert_mask()

    def _add(self, other):
        class_method = type(self)
        new_mask = class_method(self.mask)
        if isinstance(other, MaskBase):
            new_mask.mask = new_mask.mask + other.mask
            return new_mask
        if isinstance(other, np.ndarray):
            new_mask.mask = new_mask.mask + other
            return new_mask
        else:
            raise ValueError(f'Cannot do arithmetic operation with MaskBase and {type(other)}')

    def _mult(self, value):
        if not isinstance(value, numbers.Real):
            raise ValueError(f'Cannot do operation with MaskBase and {type(value)}')
        class_method = type(self)
        new_mask = class_method(self.mask)
        new_mask.mask = new_mask.mask * value
        return new_mask

    def __add__(self, other):
        return self._add(other)

    def __sub__(self, other):
        return self + (-1 * other)

    def __iadd__(self, other):
        return self + other

    def __isub__(self, other):
        return self - other

    def __mul__(self, value):
        return self._mult(value)

    def __rmul__(self, value):
        return self._mult(value)

    def __div__(self, value):
        return self._mult(1 / float(value))

    def __truediv__(self, value):
        return self.__div__(value)

    def __imul__(self, value):
        return self * value

    def __idiv__(self, value):
        return self / value

    def __itruediv__(self, value):
        return self.__idiv__(value)

    def __eq__(self, other):
        return np.array_equal(self.mask, other.mask)

    def __ne__(self, other):
        return not self.__eq__(other)
