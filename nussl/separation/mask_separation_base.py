#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base class for separation algorithms that make masks. Most algorithms in nussl are derived from MaskSeparationBase. 

"""
import json
import warnings

import masks
import separation_base
from ..core import utils
from ..core import audio_signal
from ..core import constants


class MaskSeparationBase(separation_base.SeparationBase):
    """
    Base class for separation algorithms that create a mask (binary or soft) to do their separation. Most algorithms 
    in nussl are derived from :class:`MaskSeparationBase`.
    
    Although this class will do nothing if you instantiate and run it by itself, algorithms that are derived from this
    class are expected to return a list of :class:`separation.masks.mask_base.MaskBase` -derived objects 
    (i.e., either a :class:`separation.masks.binary_mask.BinaryMask` or :class:`separation.masks.soft_mask.SoftMask`
    object) by their :func:`run()` method. Being a subclass of :class:`MaskSeparationBase` is an implicit contract 
    assuring this.  Returning a :class:`separation.masks.mask_base.MaskBase`-derived object standardizes 
    algorithm return types for :class:`evaluation.evaluation_base.EvaluationBase`-derived objects. 
    
    Args:
        input_audio_signal: (:class:`audio_signal.AudioSignal`) An :class:`audio_signal.AudioSignal` object containing 
            the mixture to be separated.
        mask_type: (str) Indicates whether to make binary or soft masks. See :attr:`mask_type` property for details.
        mask_threshold: (float) Value between [0.0, 1.0] to convert a soft mask to a binary mask. See 
            :attr:`mask_threshold` property for details.
    
    """

    BINARY_MASK = 'binary'
    """ String alias for setting this object to return :class:`separation.masks.binary_mask.BinaryMask` objects
    """

    SOFT_MASK = 'soft'
    """ String alias for setting this object to return :class:`separation.masks.soft_mask.SoftMask` objects
    """

    _valid_mask_types = [BINARY_MASK, SOFT_MASK]

    def __init__(self, input_audio_signal, mask_type=SOFT_MASK, mask_threshold=0.5):
        super(MaskSeparationBase, self).__init__(input_audio_signal=input_audio_signal)

        self._mask_type = None
        self.mask_type = mask_type
        self._mask_threshold = None
        self.mask_threshold = mask_threshold
        self.result_masks = []

    @property
    def mask_type(self):
        """
        PROPERTY
        
        This property indicates what type of mask the derived algorithm will create and be returned by :func:`run()`.
        Options are either 'soft' or 'binary'. 
        :attr:`mask_type` is usually set when initializing a :class:`MaskSeparationBase`-derived class 
        and defaults to :attr:`SOFT_MASK`.
        
        This property, though stored as a string, can be set in two ways when initializing:
        
        * First, it is possible to set this property with a string. Only ``'soft'`` and ``'binary'`` are accepted 
          (case insensitive), every other value will raise an error. When initializing with a string, two helper 
          attributes are provided: :attr:`BINARY_MASK` and :attr:`SOFT_MASK`.
        
          It is **HIGHLY** encouraged to use these, as the API may change and code that uses bare strings 
          (e.g. ``mask_type = 'soft'`` or ``mask_type = 'binary'``) for assignment might not be future-proof. 
          :attr:`BINARY_MASK`` and :attr:`SOFT_MASK` are safe aliases in case these underlying types change.
        
        * The second way to set this property is by using a class prototype of either the 
          :class:`separation.masks.binary_mask.BinaryMask` or :class:`separation.masks.soft_mask.SoftMask` class
          prototype. This is probably the most stable way to set this, and it's fairly succinct. 
          For example, ``mask_type = nussl.BinaryMask`` or ``mask_type = nussl.SoftMask`` are both perfectly valid.
        
        Though uncommon, this can be set outside of :func:`__init__()` 
        
        Examples of both methods are shown below.
        
        Returns:
            mask_type (str): Either ``'soft'`` or ``'binary'``. 
            
        Raises:
            ValueError if set invalidly.
            
        Example:
            
        .. code-block:: python
            :linenos:
    
            import nussl
            mixture_signal = nussl.AudioSignal()
                
            # Two options for determining mask upon init...
             
            # Option 1: Init with a string (BINARY_MASK is a string 'constant')
            repet_sim = nussl.RepetSim(mixture_signal, mask_type=nussl.MaskSeparationBase.BINARY_MASK)
             
            # Option 2: Init with a class type
            ola = nussl.OverlapAdd(mixture_signal, mask_type=nussl.SoftMask)
             
            # It's also possible to change these values after init by changing the `mask_type` property...
            repet_sim.mask_type = nussl.MaskSeparationBase.SOFT_MASK  # using a string
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
        PROPERTY
        
        Threshold of determining True/False if :attr:`mask_type` is :attr:`BINARY_MASK`. Some algorithms will first 
        make a soft mask and then convert that to a binary mask using this threshold parameter. All values of the 
        soft mask are between ``[0.0, 1.0]`` and as such :func:`mask_threshold` is expected to be a float between 
        ``[0.0, 1.0]``.
        
        Returns:
            mask_threshold (float): Value between ``[0.0, 1.0]`` that indicates the True/False cutoff when converting a
            soft mask to binary mask.
                                
        Raises:
            ValueError if not a float or if set outside ``[0.0, 1.0]``.

        """
        return self._mask_threshold

    @mask_threshold.setter
    def mask_threshold(self, value):
        if not isinstance(value, float) or not (0.0 < value < 1.0):
            raise ValueError('Mask threshold must be a float between [0.0, 1.0]!')

        self._mask_threshold = value

    def zeros_mask(self, shape):
        """
        Creates a new zeros mask with this object's type

        Args:
            shape:

        Returns:

        """
        if self.mask_type == self.BINARY_MASK:
            return masks.BinaryMask.zeros(shape)
        else:
            return masks.SoftMask.zeros(shape)

    def ones_mask(self, shape):
        """

        Args:
            shape:

        Returns:

        """
        if self.mask_type == self.BINARY_MASK:
            return masks.BinaryMask.ones(shape)
        else:
            return masks.SoftMask.ones(shape)

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
        """Makes :class:`audio_signal.AudioSignal` objects after mask-based separation algorithm is run. 
        Base class: Do not call directly!

        Raises:
            NotImplementedError: Cannot call base class!
        """
        raise NotImplementedError('Cannot call base class!')

    @classmethod
    def from_json(cls, json_string):
        """
        Creates a new :class:`SeparationBase` object from the parameters stored in this JSON string.

        Args:
            json_string (str): A JSON string containing all the data to create a new :class:`SeparationBase`
                object.

        Returns:
            (:class:`SeparationBase`) A new :class:`SeparationBase` object from the JSON string.

        See Also:
            :func:`to_json` to make a JSON string to freeze this object.

        """
        mask_sep_decoder = MaskSeparationBaseDecoder(cls)
        return mask_sep_decoder.decode(json_string)


class MaskSeparationBaseDecoder(separation_base.SeparationBaseDecoder):
    """ Object to decode a :class:`MaskSeparationBase`-derived object from JSON serialization.
    You should never have to instantiate this object by hand.
    """

    def __init__(self, separation_class):
        self.separation_class = separation_class
        json.JSONDecoder.__init__(self, object_hook=self._json_separation_decoder)

    def _json_separation_decoder(self, json_dict):
        if '__class__' in json_dict and '__module__' in json_dict:
            json_dict, separator = self._inspect_json_and_create_new_instance(json_dict)

            # fill out the rest of the fields
            for k, v in json_dict.items():
                if isinstance(v, dict) and constants.NUMPY_JSON_KEY in v:
                    separator.__dict__[k] = utils.json_numpy_obj_hook(v[constants.NUMPY_JSON_KEY])
                    
                # TODO: test this in python3
                elif isinstance(v, (str, bytes, unicode)) and audio_signal.__name__ in v:

                    separator.__dict__[k] = audio_signal.AudioSignal.from_json(v)
                elif k == 'result_masks':
                    # for mask_json in v:

                    separator.result_masks = [masks.MaskBase.from_json(itm) for itm in v]
                else:
                    separator.__dict__[k] = v if not isinstance(v, unicode) else v.encode('ascii')

            return separator
        else:
            return json_dict
