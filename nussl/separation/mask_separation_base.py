#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base class for separation algorithms that make masks.

"""
import warnings

import separation_base
import masks


class MaskSeparationBase(separation_base.SeparationBase):
    """
    Base class for separation algorithms that create a mask (binary or soft) to do their separation. Because python
    is dynamically typed, and therefore we cannot enforce return types for SeparationBase.run(), this class provides
    a way to 
    
    """

    BINARY_MASK = 'binary'
    SOFT_MASK = 'soft'
    _valid_mask_types = [BINARY_MASK, SOFT_MASK]

    def __init__(self, input_audio_signal, mask_type=SOFT_MASK, mask_threshold=0.5):
        super(MaskSeparationBase, self).__init__(input_audio_signal=input_audio_signal)

        self._mask_type = None
        self.mask_type = mask_type
        self._mask_threshold = None
        self.mask_threshold = mask_threshold

    @property
    def mask_type(self):
        """
        
        Returns:

        """
        return self._mask_type

    @mask_type.setter
    def mask_type(self, value):
        error = ValueError('Invalid mask type! Got {0} but valid masks are: [{1}]!'
                           .format(value, ', '.join(self._valid_mask_types)))

        if value is None:
            raise error

        if isinstance(value, str):
            value = value.lower()
            if value in self._valid_mask_types:
                self._mask_type = value
            else:
                raise error

        elif isinstance(value, masks.MaskBase):
            warnings.warn('This separation method is not using the values in the provided mask object.')
            value = type(value).__name__
            value = value[:value.find('Mask')].lower()

            if value not in self._valid_mask_types:
                # make sure we don't get duped by accident. This shouldn't happen
                raise error
            self._mask_type = value

        elif issubclass(value, masks.MaskBase):
            if value is masks.BinaryMask:
                self._mask_type = self.BINARY_MASK
            elif value is masks.SoftMask:
                self._mask_type = self.SOFT_MASK
            else:
                raise error

        else:
            raise error

    @property
    def mask_threshold(self):
        """
        
        Returns:

        """
        return self._mask_threshold

    @mask_threshold.setter
    def mask_threshold(self, value):
        if not isinstance(value, float) or not (0.0 < value and value < 1.0):
            raise ValueError('Mask threshold must be a float between [0.0, 1.0]!')

        self._mask_threshold = value
