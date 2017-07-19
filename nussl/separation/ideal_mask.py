#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
``IdealMask`` separates sources using the ideal binary or soft mask from ground truth. It accepts a list of 
``AudioSignal``
objects, each of which contains a known source, and applies a mask to a time-frequency representation of the
input mixture created from each of the known sources. This is often used as an upper bound on source separation
performance when benchmarking new algorithms, as it represents the best possible scenario for mask-based methods.

At the time of this writing, the time-frequency representation used by this class is the magnitude spectrogram

This class is derived from ``MaskSeparationBase`` so its ``run()`` method returns a list of ``MaskBase`` 
objects.
    
"""

import numpy as np
import warnings

import nussl.config
import nussl.utils
import nussl.spectral_utils
import mask_separation_base
import masks


class IdealMask(mask_separation_base.MaskSeparationBase):
    """
    Args:
        input_audio_mixture (:obj:`AudioSignal`): Input ``AudioSignal`` mixture to create the masks from.
        sources_list (list): List of ``AudioSignal`` objects where each one represents an isolated source in 
            the mixture.
        mask_type (str, Optional): Indicates whether to make a binary or soft mask. Optional, defaults to ``SOFT_MASK``
        use_librosa_stft (bool, Optional): Whether to use librosa's STFT function. Optional, defaults to config 
            settings.
        
    Attributes:
        sources (list): List of ``AudioSignal`` objects from ``__init__()`` where each object represents a single
            isolated sources within the mixture.
        estimated_masks (list): List of resultant ``MaskBase`` objects created. Masks in this list are in
            the same order that ``source_list`` (and ``self.sources``) is in.
        estimated_sources (list): List of ``AudioSignal`` objects created from applying the created masks to the
            mixture.
            
    Raises:
        ValueError: If not all items in ``sources_list`` are ``AudioSignal`` objects, OR if not all of the 
            ``AudioSignal`` objects in ``sources_list`` have the same sample rate and number of channels,
            OR if ``input_audio_mixture`` has a different sample rate or number of channels as the ``AudioSignal`` 
            objects in ``sources_list``.
            
    Example:
         ::
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

    def __init__(self, input_audio_mixture, sources_list,
                 mask_type=mask_separation_base.MaskSeparationBase.SOFT_MASK,
                 use_librosa_stft=nussl.config.USE_LIBROSA_STFT):
        super(IdealMask, self).__init__(input_audio_signal=input_audio_mixture, mask_type=mask_type)

        self.sources = nussl.utils._verify_audio_signal_list_strict(sources_list)

        # Make sure input_audio_signal has the same settings as sources_list
        if self.audio_signal.sample_rate != self.sources[0].sample_rate:
            raise ValueError('input_audio_signal must have the same sample rate as entries of sources_list!')
        if self.audio_signal.num_channels != self.sources[0].num_channels:
            raise ValueError('input_audio_signal must have the same number of channels as entries of sources_list!')

        self.estimated_masks = None
        self.estimated_sources = None
        self.use_librosa_stft = use_librosa_stft

    def run(self):
        """
        Creates a list of masks (as ``MaskBase`` objects, either ``BinaryMask`` or ``SoftMask`` depending 
        on how the object was instantiated) from a list of known source signals (``source_list`` in the constructor).
        Returns a list of ``MaskBase`` objects (one for each input signal) in the order that they were provided
        when this object was initialized.
        
        Binary masks are created based on the magnitude spectrogram using the following formula:
        
                ``mask = (provided_source.mag_spec >= (mixture_mag_spec - provided_source.mag_spec)``
                
        Where ``-`` is a element-wise subtraction (as if the values were binary ints, 0 or 1) and ``>=``
        is element-wise logical greater-than-or-equal (again, as if the values were binary ints, 0 or 1).
        
        
        Soft masks are also created based on the magnitude spectrogram but use the following formula:
        
                1) ``mask = mixture_mag_spec / provided_source.mag_spec``
                
                2) ``mask = log(mask)``
                
                3) ``mask = (mask + abs(min(mask))) / max(mask)``
                
        
        Where all arithmetic operations and log are element-wise. This provides a logarithmically scaled mask that is
        in the interval [0.0, 1.0].
        
        
        Returns:
            estimated_masks (list): List of resultant ``MaskBase`` objects created. Masks in this list are in
            the same order that ``source_list`` (and ``self.sources``) are in.
                
        Raises:
            RuntimeError if unknown mask type is provided (Options are [``BinaryMask``, or ``SoftMask``]).

        """
        self._compute_spectrograms()
        self.estimated_masks = []

        for source in self.sources:
            if self.mask_type == self.BINARY_MASK:
                mag = source.magnitude_spectrogram_data  # Alias this variable, for easy reading
                cur_mask = (mag >= (self._mixture_mag_spec - mag))
                mask = masks.BinaryMask(cur_mask)

            elif self.mask_type == self.SOFT_MASK:
                # TODO: This is a kludge. What is the actual right way to do this?
                sm = np.divide(self.audio_signal.magnitude_spectrogram_data, source.magnitude_spectrogram_data)
                # log_sm1 = np.log(sm - np.min(sm) + 1)
                log_sm = np.log(sm)
                log_sm += np.abs(np.min(log_sm))
                log_sm /= np.max(log_sm)
                mask = masks.SoftMask(sm)
            else:
                raise RuntimeError('Unknown mask type: {}'.format(self.mask_type))

            self.estimated_masks.append(mask)

        return self.estimated_masks

    @property
    def residual(self):
        """
        This is an ``AudioSignal`` object that contains the left over audio that was not captured 
        by creating the masks. The residual is calculated in the time domain; after all of the masks are
        created by running ``IdealMask.run()`` and making the corresponding ``AudioSignal`` objects 
        (using ``IdealMask.make_audio_signals()`` which applies the masks to the mixture stft and does an istft for
        each source from the calculated masks), the residual is simply the original mixture with 
        
        Returns:
            residual (:obj:`AudioSignal`): ``AudioSignal`` object that contains the left over 
            audio that was not captured by creating the masks.
        
        Raises:
            ValueError if ``IdealMask.run()`` has not been called. OR
            Exception if there was an unforeseen issue.

        """
        if self.estimated_masks is None:
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
        """Returns a list of signals (as ``AudioSignal`` objects) created by applying the ideal masks.
        This creates the signals by element-wise multiply the masks with the mixture stft.
        Prior to running this, it is expected that ``run()`` has been called or else this will throw an error.
        These of signals is in the same order as the input ideal mixtures were when they were input (as a parameter
        to the constructor, ``sources_list``).

        Returns:
            estimated_sources (list): List of ``AudioSignal`` objects that represent the sources created by applying
            a mask from the known source to the mixture

        Example:
             :: 
            ideal_mask = nussl.IdealMask(mixture, [drums, flute], mask_type=nussl.BinaryMask)
            ideal_mask.run()
            ideal_drums, ideal_flute = ideal_mask.make_audio_signals()
            
        """
        if self.estimated_masks is None or self.audio_signal.stft_data.size <= 0:
            raise ValueError('Cannot make audio signals prior to running algorithm!')

        self.estimated_sources = []

        for cur_mask in self.estimated_masks:
            estimated_stft = np.multiply(cur_mask.mask, self.audio_signal.stft_data)
            new_signal = self.audio_signal.make_copy_with_stft_data(estimated_stft, verbose=False)
            new_signal.istft(self.stft_params.window_length, self.stft_params.hop_length,
                             self.stft_params.window_type, overwrite=True,
                             use_librosa=self.use_librosa_stft,
                             truncate_to_length=self.audio_signal.signal_length)

            self.estimated_sources.append(new_signal)

        return self.estimated_sources
