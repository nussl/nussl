#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The :class:`BinaryMask` class is for creating a time-frequency mask with binary values. Like all 
:class:`separation.masks.mask_base.MaskBase` objects, :class:`BinaryMask` is initialized with a 2D or 3D numpy array
containing the mask data. The data type (numpy.dtype) of the initial mask can be either bool, int, or float. 
The mask is stored as a 3-dimensional boolean-valued numpy array.

The best case scenario for the input mask np array is when the data type is bool. If the data type of the input mask 
upon init is int it is expected that all values are either 0 or 1. If the data type
of the mask is float, all values must be within 1e-2 of either 1 or 0. If the array is not set as one of these, 
:class:`BinaryMask` will raise an exception.

:class:`BinaryMask` (like :class:`separation.masks.soft_mask.SoftMask`) is one of the return types for the :func:`run()` 
methods of :class:`separation.mask_separation_base.MaskSeparationBase`-derived objects (this is most of the 
separation methods in `nussl`.

See Also:
    * :class:`separation.masks.mask_base.MaskBase`: The base class for BinaryMask and SoftMask
    * :class:`separation.masks.soft_mask.SoftMask`: Similar to BinaryMask, but instead of taking boolean values, 
      takes floats between ``[0.0 and 1.0]``.
    * :class:`separation.mask_separation_base.MaskSeparationBase`: Base class for all mask-based separation methods 
      in `nussl`.

Examples:
    Initializing a mask from a numpy array...
    
.. code-block:: python
    :linenos:
    
    import nussl
    import numpy as np
    
    # load a file
    signal = nussl.AudioSignal('path/to/file.wav')
    stft = signal.stft()
    
    # Make a random binary mask with the same shape as the stft with dtype == bool
    rand_bool_mask = np.random.randint(2, size=stft.shape).astype('bool')
    bin_mask_bool = nussl.BinaryMask(rand_bool_mask)
    
    # Make a random binary mask with the same shape as the stft with dtype == int
    rand_int_mask = np.random.randint(2, size=stft.shape)
    bin_mask_int = nussl.BinaryMask(rand_int_mask)
    
    # Make a random binary mask with the same shape as the stft with dtype == float
    rand_float_mask = np.random.randint(2, size=stft.shape).astype('float')
    bin_mask_int = nussl.BinaryMask(rand_float_mask)    
    
    
:class:`separation.mask_separation_base.MaskSeparationBase`-derived methods return 
:class:`separation.masks.mask_base.MaskBase` masks, like so...

.. code-block:: python
    :linenos:

    import nussl
    
    # load a file
    signal = nussl.AudioSignal('path/to/file.wav')
    
    repet = nussl.Repet(signal, mask_type=nussl.BinaryMask)  # You have to specify that you want Binary Masks back
    assert isinstance(repet, nussl.MaskSeparationBase)  # Repet is a MaskSeparationBase-derived class
    
    [background_mask, foreground_mask] = repet.run()  # MaskSeparationBase-derived classes return MaskBase objects
    assert isinstance(foreground_mask, nussl.BinaryMask)  # this is True
    assert isinstance(background_mask, nussl.BinaryMask)  # this is True

"""

import numpy as np

import mask_base


class BinaryMask(mask_base.MaskBase):
    """
    Class for creating a Binary Mask to apply to a time-frequency representation of the audio. 
    
    Args:
        input_mask (:obj:`np.ndarray`): 2- or 3-D :obj:`np.array` that represents the mask.
    """

    def __init__(self, input_mask=None, mask_shape=None):
        super(BinaryMask, self).__init__(input_mask, mask_shape)

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
        Returns this :class:`BinaryMask` as a numpy array of ints of 0's and 1's.
        
        Returns:
            numpy :obj:`ndarray` of this :obj:`BinaryMask` represented as ints instead of bools.

        """
        if channel is None:
            return self.mask.astype('int')
        else:
            return self.get_channel(channel).astype('int')

    def invert_mask(self):
        """
        Makes a new :class:`BinaryMask` object with a logical not applied to flip the values in this :class:`BinaryMask`
        object.

        Returns:
            A new :class:`BinaryMask` object that has all of the boolean values flipped.

        """
        return BinaryMask(np.logical_not(self.mask))

    @staticmethod
    def mask_to_binary(mask_, threshold):
        """
        Makes a binary mask from a soft mask with a True/False threshold.
        
        Args:
            mask_ (:obj:`MaskBase` or :obj:`np.ndarray`): Soft mask to convert to :class:`BinaryMask`
            threshold (float): Value between ``[0.0, 1.0]`` to determine the True/False cutoff

        Returns:

        """
        if isinstance(mask_, np.ndarray):
            return mask_ > threshold
        elif isinstance(mask_, mask_base.MaskBase):
            return mask_.mask > threshold
