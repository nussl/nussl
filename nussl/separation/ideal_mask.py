#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:class:`IdealMask` separates sources using the ideal binary or soft mask from ground truth. It accepts a list of 
:class:`audio_signal.AudioSignal` objects, each of which contains a known source, and applies a mask to a 
time-frequency representation of the input mixture created from each of the known sources. This is often used as
an upper bound on source separation performance when benchmarking new algorithms, as it represents the best possible
scenario for mask-based methods.

At the time of this writing, the time-frequency representation used by this class is the magnitude spectrogram.

This class is derived from :class:`separation.mask_separation_base.MaskSeparationBase` so its 
:func:`run()` method returns a list of :class:`separation.masks.mask_base.MaskBase` objects.
    
"""

import warnings

import numpy as np
import librosa

import mask_separation_base
import masks
from ..core import constants
from ..core import utils


class IdealMask(mask_separation_base.MaskSeparationBase):
    """
    Args:
        input_audio_mixture (:class:`audio_signal.AudioSignal`): Input :class:`audio_signal.AudioSignal` mixture to 
            create the masks from.
        sources_list (list): List of :class:`audio_signal.AudioSignal` objects where each one represents an 
            isolated source in the mixture.
        mask_type (str, Optional): Indicates whether to make a binary or soft mask. Optional, defaults to 
            :attr:`SOFT_MASK`.
        use_librosa_stft (bool, Optional): Whether to use librosa's STFT function. Optional, defaults to config 
            settings.
        
    Attributes:
        sources (list): List of :class:`audio_signal.AudioSignal` objects from :func:`__init__()` where each object 
            represents a single isolated sources within the mixture.
        estimated_masks (list): List of resultant :class:`separation.masks.mask_base.MaskBase` objects created. 
            Masks in this list are in the same order that ``source_list`` (and :attr:`sources`) is in.
        estimated_sources (list): List of :class:`audio_signal.AudioSignal` objects created from applying the 
            created masks to the mixture.
            
    Raises:
        ValueError: If not all items in ``sources_list`` are :class:`audio_signal.AudioSignal` objects, OR if not 
            all of the  :class:`audio_signal.AudioSignal` objects in ``sources_list`` have the same sample rate and 
            number of channels, OR if ``input_audio_mixture`` has a different sample rate or number of channels as the 
            :class:`audio_signal.AudioSignal`  objects in ``sources_list``.
            
    Example:
        
    .. code-block:: python
        :linenos:

        import nussl
        import os
        
        path_to_drums = os.path.join('demo_files', 'drums.wav')
        path_to_flute = os.path.join('demo_files', 'flute.wav')

        drums = nussl.AudioSignal(path_to_drums)
        drums.to_mono(overwrite=True)  # make it mono
        
        flute = nussl.AudioSignal(path_to_flute)
        flute.truncate_samples(drums.signal_length)  # shorten the flute solo
        
        # Make a mixture and increase the gain on the flute
        mixture = drums + flute * 3.0
        
        # Run IdealMask making binary masks
        ideal_mask = nussl.IdealMask(mixture, [drums, flute], mask_type=nussl.BinaryMask)
        ideal_mask.run()
        ideal_drums, ideal_flute = ideal_mask.make_audio_signals()
        ideal_residual = ideal_mask.residual  # Left over audio that was not captured by the mask
        
    """

    def __init__(self, input_audio_mixture, sources_list, power=1, split_zeros=False, binary_db_threshold=20,
                 mask_type=mask_separation_base.MaskSeparationBase.SOFT_MASK,
                 use_librosa_stft=constants.USE_LIBROSA_STFT):
        super(IdealMask, self).__init__(input_audio_signal=input_audio_mixture, mask_type=mask_type)

        self.sources = utils.verify_audio_signal_list_strict(sources_list)

        # Make sure input_audio_signal has the same settings as sources_list
        if self.audio_signal.sample_rate != self.sources[0].sample_rate:
            raise ValueError('input_audio_signal must have the same sample rate as entries of sources_list!')
        if self.audio_signal.num_channels != self.sources[0].num_channels:
            raise ValueError('input_audio_signal must have the same number of channels as entries of sources_list!')

        self.result_masks = None
        self.estimated_sources = None
        self._mixture_mag_spec = None
        self.use_librosa_stft = use_librosa_stft

        self.power = power
        self.split_zeros = split_zeros
        self.binary_db_threshold = binary_db_threshold

    def run(self):
        """
        Creates a list of masks (as :class:`separation.masks.mask_base.MaskBase` objects, either 
        :class:`separation.masks.binary_mask.BinaryMask` or :class:`separation.masks.soft_mask.SoftMask` 
        depending on how the object was instantiated) from a list of known source signals (``source_list`` 
        in the constructor).
        
        Returns a list of :class:`separation.masks.mask_base.MaskBase` objects (one for each input signal) 
        in the order that they were provided when this object was initialized.
        
        Binary masks are created based on the magnitude spectrogram using the following formula:
        
                ``mask = (provided_source.mag_spec >= (mixture_mag_spec - provided_source.mag_spec)``
                ``mask = (20 * np.log10(source.mag_spec / mixture.mag_spec)) > binary_db_threshold``

        Where '``/``' is a element-wise division and '``>``' is element-wise logical greater-than.
        
        
        Soft masks are also created based on the magnitude spectrogram but use the following formula:
        
                1) ``mask = mixture_mag_spec / provided_source.mag_spec``
                
                2) ``mask = log(mask)``
                
                3) ``mask = (mask + abs(min(mask))) / max(mask)``
                
        
        Where all arithmetic operations and log are element-wise. This provides a logarithmically scaled mask that is
        in the interval [0.0, 1.0].
        
        
        Returns:
            estimated_masks (list): List of resultant :class:`separation.masks.mask_base.MaskBase` objects created. 
            Masks in this list are in the same order that ``source_list`` (and :attr:`sources`) are in.
                
        Raises:
            RuntimeError if unknown mask type is provided (Options are [``BinaryMask``, or ``SoftMask``]).

        """
        self._compute_spectrograms()
        self.result_masks = []

        for source in self.sources:
            mag = source.magnitude_spectrogram_data # Alias this variable, for easy reading
            if self.mask_type == self.BINARY_MASK:
                div = np.divide(mag + constants.EPSILON, self._mixture_mag_spec + constants.EPSILON)
                cur_mask = (20 * np.log10(div)) > self.binary_db_threshold
                mask = masks.BinaryMask(cur_mask)

            elif self.mask_type == self.SOFT_MASK:
                soft_mask = librosa.util.softmask(self.audio_signal.magnitude_spectrogram_data,
                                                  mag, power=self.power,
                                                  split_zeros=self.split_zeros)

                mask = masks.SoftMask(soft_mask)
            else:
                raise RuntimeError('Unknown mask type: {}'.format(self.mask_type))

            self.result_masks.append(mask)

        return self.result_masks

    @property
    def residual(self):
        """
        This is an :class:`audio_signal.AudioSignal` object that contains the left over audio that was not captured 
        by creating the masks. The residual is calculated in the time domain; after all of the masks are
        created by running :func:`run()` and making the corresponding :class:`audio_signal.AudioSignal` objects 
        (using :func:`make_audio_signals()` which applies the masks to the mixture stft and does an istft for
        each source from the calculated masks), the residual is simply the original mixture with 
        
        Returns:
            residual (:class:`audio_signal.AudioSignal`): :class:`audio_signal.AudioSignal` object that contains the 
            left over audio that was not captured by creating the masks.
        
        Raises:
            * ValueError if :func:`run()` has not been called. OR
            * Exception if there was an unforeseen issue.

        """
        if self.result_masks is None:
            raise ValueError('Cannot calculate residual prior to running algorithm!')

        if self.estimated_sources is None:
            warnings.warn('Need to run self.make_audio_signals prior to calculating residual...')
            self.make_audio_signals()
        else:
            residual = self.audio_signal
            for source in self.estimated_sources:
                residual = residual - source

            return residual

        raise Exception('Could not make residual!')

    def _compute_spectrograms(self):
        self.audio_signal.stft(overwrite=True, remove_reflection=True, use_librosa=self.use_librosa_stft)

        # Alias this variable for ease
        self._mixture_mag_spec = self.audio_signal.magnitude_spectrogram_data
        for source in self.sources:
            source.stft(overwrite=True, remove_reflection=True, use_librosa=self.use_librosa_stft)

    def make_audio_signals(self):
        """Returns a list of signals (as :class:`audio_signal.AudioSignal` objects) created by applying the ideal masks.
        This creates the signals by element-wise multiply the masks with the mixture stft.
        Prior to running this, it is expected that :func:`run()` has been called or else this will throw an error.
        These of signals is in the same order as the input ideal mixtures were when they were input (as a parameter
        to the constructor, ``sources_list`` or the :attr:`sources` attribute).

        Returns:
            estimated_sources (list): List of :class:`audio_signal.AudioSignal` objects that represent the sources 
            created by applying a mask from the known source to the mixture

        Example:

        .. code-block:: python
            :linenos:
            
            ideal_mask = nussl.IdealMask(mixture, [drums, flute], mask_type=nussl.BinaryMask)
            ideal_mask.run()
            ideal_drums, ideal_flute = ideal_mask.make_audio_signals()
            
        """
        if self.result_masks is None or self.audio_signal.stft_data.size <= 0:
            raise ValueError('Cannot make audio signals prior to running algorithm!')

        self.estimated_sources = []

        for cur_mask in self.result_masks:
            estimated_stft = np.multiply(cur_mask.mask, self.audio_signal.stft_data)
            new_signal = self.audio_signal.make_copy_with_stft_data(estimated_stft, verbose=False)
            new_signal.istft(self.stft_params.window_length, self.stft_params.hop_length,
                             self.stft_params.window_type, overwrite=True,
                             use_librosa=self.use_librosa_stft,
                             truncate_to_length=self.audio_signal.signal_length)

            self.estimated_sources.append(new_signal)

        return self.estimated_sources
