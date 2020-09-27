"""
Base class for separation algorithms that make masks. Most algorithms in 
nussl are derived from MaskSeparationBase. 
"""

from ...core import masks
from . import SeparationBase
from .separation_base import SeparationException


class MaskSeparationBase(SeparationBase):
    """
    Base class for separation algorithms that create a mask (binary or soft) to do 
    their separation. Most algorithms in nussl are derived from 
    :class:`MaskSeparationBase`.
    
    Although this class will do nothing if you instantiate and run it by itself,
    algorithms that are derived from this class are expected to return a list of 
    :class:`separation.masks.mask_base.MaskBase` -derived objects  
    (i.e., either a :class:`separation.masks.binary_mask.BinaryMask` or 
    :class:`separation.masks.soft_mask.SoftMask` object) by their :func:`run()` 
    method. Being a subclass of :class:`MaskSeparationBase` is an implicit contract 
    assuring this.  Returning a :class:`separation.masks.mask_base.MaskBase`-derived 
    object standardizes algorithm return types for 
    :class:`evaluation.evaluation_base.EvaluationBase`-derived objects. 
    
    Args:
        input_audio_signal: (:class:`audio_signal.AudioSignal`) An 
          :class:`audio_signal.AudioSignal` object containing the mixture to be 
          separated.
        mask_type: (str, BinaryMask, or SoftMask) Indicates whether to make 
          binary or soft masks. See :attr:`mask_type` property for details.
        mask_threshold: (float) Value between [0.0, 1.0] to convert a soft mask 
          to a binary mask. See :attr:`mask_threshold` property for details.
    """
    
    MASKS = {
        'binary': masks.BinaryMask,
        'soft': masks.SoftMask
    }

    def __init__(self, input_audio_signal, mask_type='soft', mask_threshold=0.5):
        super().__init__(input_audio_signal=input_audio_signal)

        self.mask_type = mask_type
        self.mask_threshold = mask_threshold
        self.result_masks = []

        self.metadata.update({
            'mask_type': mask_type,
            'mask_threshold': 'N/A' if self.mask_type == masks.SoftMask else mask_threshold
        })

    @property
    def mask_type(self):
        """        
        This property indicates what type of mask the derived algorithm will create 
        and be returned by :func:`run()`. Options are either 'soft' or 'binary'. 
        :attr:`mask_type` is usually set when initializing a 
        :class:`MaskSeparationBase`-derived class and defaults to 'soft..
        
        This property, though stored as a string, can be set in two ways when 
        initializing:
        
        * First, it is possible to set this property with a string. Only ``'soft'`` 
          and ``'binary'`` are accepted (case insensitive), every other value will 
          raise an error. When initializing with a string, two helper 
          attributes are provided: :attr:`BINARY_MASK` and :attr:`SOFT_MASK`.
        
          It is **HIGHLY** encouraged to use these, as the API may change and code 
          that uses bare strings (e.g. ``mask_type = 'soft'`` or 
          ``mask_type = 'binary'``) for assignment might not be future-proof. 
          :attr:`BINARY_MASK`` and :attr:`SOFT_MASK` are safe aliases in case these 
          underlying types change.
        
        * The second way to set this property is by using a class prototype of 
          either the :class:`separation.masks.binary_mask.BinaryMask` or 
          :class:`separation.masks.soft_mask.SoftMask` class prototype. This is 
          probably the most stable way to set this, and it's fairly succinct. 
          For example, ``mask_type = nussl.BinaryMask`` or 
          ``mask_type = nussl.SoftMask`` are both perfectly valid.
        
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
        error = ValueError(
            f"Invalid mask type! Got {value} but valid masks are:"
            f" [{', '.join(list(self.MASKS.keys()))}]!"
        )

        if value is None:
            raise error

        if isinstance(value, str):
            value = value.lower()
            if value in self.MASKS:
                self._mask_type = self.MASKS[value]
            else:
                raise error
        elif isinstance(value, masks.BinaryMask):
            self._mask_type = self.MASKS['binary']
        elif isinstance(value, masks.SoftMask):
            self._mask_type = self.MASKS['soft']
        else:
            raise error

    @property
    def mask_threshold(self):
        """        
        Threshold of determining True/False if :attr:`mask_type` is 
        :attr:`BINARY_MASK`. Some algorithms will first make a soft mask and then 
        convert that to a binary mask using this threshold parameter. All 
        values of the soft mask are between ``[0.0, 1.0]`` and as such 
        :func:`mask_threshold` is expected to be a float between 
        ``[0.0, 1.0]``.
        
        Returns:
            mask_threshold (float): Value between ``[0.0, 1.0]`` that indicates 
              the True/False cutoff when converting a soft mask to binary mask.
                                
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
        Creates a new zeros mask with this object's type.

        Args:
            shape (tuple): tuple with shape of mask

        Returns:
            A subclass of `MaskBase` containing 0s.
        """
        return self.mask_type.zeros(shape)

    def ones_mask(self, shape):
        """
        Creates a new ones mask with this object's type.

        Args:
            shape (tuple): tuple with shape of mask

        Returns:
            A subclass of `MaskBase` containing 1s.
        """
        return self.mask_type.ones(shape)

    def _preprocess_audio_signal(self):
        """
        Masking based separation algorithm always need an STFT to work with. 
        So here, the STFT of the AudioSignal object belonging to this separation
        algorithm is taken. It also resets the `self.result_masks` object to
        an empty list - new audio signal means new masks.

        This gets called when the `self.audio_signal` is set.
        """
        self.stft = self.audio_signal.stft()
        self.result_masks = []

    def run(self):
        """Runs mask-based separation algorithm. Base class: Do not call directly!

        Raises:
            NotImplementedError: Cannot call base class!
        """
        raise NotImplementedError('Cannot call base class!')

    def make_audio_signals(self):
        """
        Makes :class:`audio_signal.AudioSignal` objects after mask-based
        separation algorithm is run. This looks in ``self.result_masks``
        which must be filled by ``run`` in the algorithm that
        subclasses this. It applies each mask to the mixture audio 
        signal and returns a list of the estimates, which are each
        AudioSignal objects.

        Returns:
            list: List of AudioSignal objects corresponding to the 
              separated estimates.
        """
        if not self.result_masks:
            raise SeparationException(
                "self.result_masks is empty! Did you call self.run()?")
        
        estimates = []
        for mask in self.result_masks:
            if not isinstance(mask, self.mask_type):
                raise SeparationException(
                    f"Expected {self.mask_type} but got {type(mask)} "
                    f"in self.result_masks!"
                )
            estimate = self.audio_signal.apply_mask(mask, overwrite=False)
            estimate.istft()
            estimates.append(estimate)
        return estimates
