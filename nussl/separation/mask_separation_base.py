#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base class for separation algorithms that make masks. Most algorithms in nussl are derived from MaskSeparationBase. 

"""
import warnings

import separation_base
import masks


class MaskSeparationBase(separation_base.SeparationBase):
    """
    Base class for separation algorithms that create a mask (binary or soft) to do their separation. Most algorithms 
    in nussl are derived from ``MaskSeparationBase``.
    
    Although this class will do nothing if you instantiate and run it by itself, algorithms that are derived from this
    class are expected to return a list of :ref:`mask_base` -derived objects (i.e., either a :ref:`BinaryMask` or 
    ``SoftMask`` object)
    by their ``run()`` method. Being a subclass of ``MaskSeparationBase`` is an implicit contract assuring this. 
    Returning a ``MaskBase``-derived object standardizes algorithm return types for ``EvaluationBase`` -derived objects. 
    
    Args:
        input_audio_signal: (:obj:`AudioSignal`) An ``AudioSignal`` object containing the mixture to be separated.
        mask_type: (str) Indicates whether to make binary or soft masks. See ``mask_type`` property for details.
        mask_threshold: (float) Value between [0.0, 1.0] to convert a soft mask to a binary mask. See ``mask_threshold``
            property for details.
    
    """

    BINARY_MASK = 'binary'  #: String alias for setting this object to return ``BinaryMask`` objects
    SOFT_MASK = 'soft'  #: String alias for setting this object to return ``SoftMask`` objects
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
        This property indicates what type of mask the derived algorithm will create and be returned by `run()`.
        Options are either 'soft' or 'binary'. 
        ``mask_type`` is usually set when initializing a ``MaskSeparationBase``-derived class 
        and defaults to ``SOFT_MASK``
        .
        This property, though stored as a string, can be set in two ways when initializing:
        
        First, it is possible to set this property
        with a string. Only ``'soft'`` and ``'binary'`` are accepted (case insensitive), every other value will raise an
        error. When initializing with a string, two helper attributes are provided: ``BINARY_MASK`` and 
        ``SOFT_MASK``.
        
        It is HIGHLY encouraged to use these, as the API may change and code that uses bare strings 
        (e.g. ``mask_type = 'soft'`` or ``mask_type = 'binary'``) for assignment might not be future-proof. 
        ``BINARY_MASK`` and ``SOFT_MASK`` are safe aliases in case these underlying types change.
        
        The second way to set this property is by using a class prototype of either the ``SoftMask`` or 
        ``BinaryMask`` class
        prototype. This is probably the most stable way to set this, and it's fairly succinct. 
        For example, ``mask_type = nussl.BinaryMask`` or ``mask_type = nussl.SoftMask`` are both perfectly valid.
        
        Though uncommon, this can be set outside of ``__init__()`` 
        
        Examples of both methods are shown below.
        
        Returns:
            mask_type (str): Either 'soft' or 'binary'. 
            
        Raises:
            ValueError if set invalidly.
            
        Example:
        ::
         import nussl
         mixture_signal = nussl.AudioSignal()
            
         # Two options for determining mask upon init...
         
         # Option 1: Init with a string (BINARY_MASK is a string 'constant')
         projet = nussl.Projet(mixture_signal, mask_type=nussl.MaskSeparationBase.BINARY_MASK)
         
         # Option 2: Init with a class type
         ola = nussl.OverlapAdd(mixture_signal, mask_type=nussl.SoftMask)
         
         # It's also possible to change these values after init by changing the `mask_type` property...
         projet.mask_type = nussl.MaskSeparationBase.SOFT_MASK  # using a string
         ola.mask_type = nussl.BinaryMask  # or using a class type
        

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
        Threshold of determining True/False if ``self.mask_type`` is ``BINARY_MASK``. Some algorithms will first make a 
        soft mask and then convert that to a binary mask using this threshold parameter. All values of the soft mask
        are between [0.0, 1.0] and as such ``self.mask_threshold`` is expected to be a float between [0.0, 1.0].
        
        Returns:
            mask_threshold (float): Value between [0.0, 1.0] that indicates the True/False cutoff when converting a soft
            mask to binary mask.
                                
        Raises:
            ValueError if not a float or if set outside [0.0, 1.0].

        """
        return self._mask_threshold

    @mask_threshold.setter
    def mask_threshold(self, value):
        if not isinstance(value, float) or not (0.0 < value and value < 1.0):
            raise ValueError('Mask threshold must be a float between [0.0, 1.0]!')

        self._mask_threshold = value

    def plot(self, output_name, **kwargs):
        """Plots relevant data for mask-based separation algorithm. Base class: Do not call directly!

        Raises:
            NotImplementedError: Cannot call base class!
        """
        raise NotImplementedError('Cannot call base class!')

    def run(self):
        """Runs mask-based separation algorithm. Base class: Do not call directly!

        Raises:
            NotImplementedError: Cannot call base class!
        """
        raise NotImplementedError('Cannot call base class!')

    def make_audio_signals(self):
        """Makes ``AudioSignal`` objects after mask-based separation algorithm is run. Base class: Do not call directly!

        Raises:
            NotImplementedError: Cannot call base class!
        """
        raise NotImplementedError('Cannot call base class!')
