#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The :class:`SoftMask` class is for creating a time-frequency mask with values in the range ``[0.0, 1.0]``. Like all 
:class:`separation.masks.mask_base.MaskBase` objects, :class:`SoftMask` is initialized with a 2D or 3D numpy array
containing the mask data. The data type (numpy.dtype) of the initial mask must be float. 
The mask is stored as a 3-dimensional boolean-valued numpy array.

:class:`SoftMask` (like :class:`separation.masks.soft_mask.BinaryMask`) is one of the return types for the :func:`run()` 
methods of :class:`separation.mask_separation_base.MaskSeparationBase`-derived objects (this is most of the 
separation methods in `nussl`.

See Also:
    * :class:`separation.masks.mask_base.MaskBase`: The base class for BinaryMask and SoftMask
    * :class:`separation.masks.soft_mask.BinaryMask`: Similar to BinaryMask, but instead of taking floats, 
      it accepts boolean values.
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
    
    # Make a random soft mask with the same shape as the stft
    rand_bool_mask = np.random.random(size=stft.shape)
    bin_mask_bool = nussl.SoftMask(rand_bool_mask)
    
    
:class:`separation.mask_separation_base.MaskSeparationBase`-derived methods return 
:class:`separation.masks.mask_base.MaskBase` masks, like so...

.. code-block:: python
    :linenos:

    import nussl
    
    # load a file
    signal = nussl.AudioSignal('path/to/file.wav')
    
    repet = nussl.Repet(signal, mask_type=nussl.SoftMask)  # You have to specify that you want Binary Masks back
    assert isinstance(repet, nussl.MaskSeparationBase)  # Repet is a MaskSeparationBase-derived class
    
    [background_mask, foreground_mask] = repet.run()  # MaskSeparationBase-derived classes return MaskBase objects
    assert isinstance(foreground_mask, nussl.SoftMask)  # this is True
    assert isinstance(background_mask, nussl.SoftMask)  # this is True
"""

import numpy as np
import warnings

import mask_base
import binary_mask


class SoftMask(mask_base.MaskBase):
    """
    A simple class for making a soft mask. The soft mask is represented as a numpy array of floats between
    0.0 and 1.0, inclusive. 
    
    Args:
        input_mask (:obj:`np.ndarray`): 2- or 3-D :obj:`np.array` that represents the mask.
    """

    def __init__(self, input_mask=None, mask_shape=None):
        super(SoftMask, self).__init__(input_mask, mask_shape)

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
        Create a new :class:`separation.masks.soft_mask.BinaryMask` object from this object's data.
        
        Args:
            threshold (float, Optional): Threshold (between ``[0.0, 1.0]``) to set the True/False cutoff for the binary
             mask.

        Returns:
            A new :class:`separation.masks.soft_mask.BinaryMask` object

        """
        return binary_mask.BinaryMask(self.mask > threshold)

    def invert_mask(self):
        """
        Returns a new mask with inverted values set like ``1 - mask`` for :attr:`mask`.

        Returns:
            A new :class:`SoftMask` object with values set at ``1 - mask``.

        """
        return SoftMask(np.abs(1 - self.mask))
