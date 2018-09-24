#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import mask_separation_base
import masks
from ..core import constants
import librosa
import copy


class HPSS(mask_separation_base.MaskSeparationBase):
    """Implements harmonic/percussive source separation based on:
    
    1. Fitzgerald, Derry. “Harmonic/percussive separation using median filtering.” 
    13th International Conference on Digital Audio Effects (DAFX10), Graz, Austria, 2010.
    
    2. Driedger, Müller, Disch. “Extending harmonic-percussive separation of audio.” 
    15th International Society for Music Information Retrieval Conference (ISMIR 2014),
    Taipei, Taiwan, 2014.
    
    This is a wrapper around the librosa implementation.

    Parameters:
        input_audio_signal: (AudioSignal object) The AudioSignal object that has the
                            audio data that REPET will be run on.
        kernel_size: int or tuple (kernel_harmonic, kernel_percussive) kernel size(s) for the
            median filters.
        do_mono: (Optional) (bool) Flattens AudioSignal to mono before running the algorithm
            (does not effect the input AudioSignal object)
        use_librosa_stft: (Optional) (bool) Calls librosa's stft function instead of nussl's

    """
    def __init__(self, input_audio_signal, kernel_size=31,
                 do_mono=False, use_librosa_stft=constants.USE_LIBROSA_STFT,
                 mask_type=mask_separation_base.MaskSeparationBase.SOFT_MASK):
        super(HPSS, self).__init__(input_audio_signal=input_audio_signal, mask_type=mask_type)
        self.harmonic = None
        self.percussive = None
        self.use_librosa_stft = use_librosa_stft
        self.kernel_size = kernel_size
        self.stft = None
        self.masks = None

        if do_mono:
            self.audio_signal.to_mono(overwrite=True)

    def run(self):
        """

        Returns:

        Example:
             ::

        """
        self._compute_spectrograms()

        # separate the mixture background by masking
        harmonic_masks = []
        percussive_masks = []
        for i in range(self.audio_signal.num_channels):
            # apply mask
            harmonic_mask, percussive_mask = librosa.decompose.hpss(self.stft[:, :, i],
                                                                    kernel_size=self.kernel_size,
                                                                    mask=True)
            harmonic_masks.append(harmonic_mask)
            percussive_masks.append(percussive_mask)

        # make a new audio signal for the background

        # make a mask and return
        harmonic_mask = np.array(harmonic_masks).transpose((1, 2, 0))
        percussive_mask = np.array(percussive_masks).transpose((1, 2, 0))
        both_masks = [harmonic_mask, percussive_mask]
        
        self.masks = []
        
        for mask in both_masks:
            if self.mask_type == self.BINARY_MASK:
                mask = np.round(mask)
                mask_object = masks.BinaryMask(mask)
            elif self.mask_type == self.SOFT_MASK:
                mask_object = masks.SoftMask(mask)
            else:
                raise ValueError('Unknown mask type {}!'.format(self.mask_type))
            self.masks.append(mask_object)
        return self.masks
    
    def _compute_spectrograms(self):
        self.stft = self.audio_signal.stft(overwrite=True, remove_reflection=True,
                                           use_librosa=self.use_librosa_stft)

    def make_audio_signals(self):
        """ Returns the background and foreground audio signals. You must have run :func:`run()` prior
        to calling this function. This function will return ``None`` if :func:`run()` has not been
        called.
        
        Order of the list is ``[self.background, self.foreground]`` 

        Returns:
            (list): List containing two :class:`audio_signal.AudioSignal` objects, one for the
            calculated background
            and the next for the remaining foreground, in that order.

        Example:
            
        .. code-block:: python
            :linenos:
            
            # set up AudioSignal object
            signal = nussl.AudioSignal('path_to_file.wav')

            # set up and run repet
            hpss = nussl.HPSS(signal)
            hpss.run()

            # get audio signals (AudioSignal objects)
            harmonic, percussive = ft2d.make_audio_signals()
            
        """
        self.sources = []
        for mask in self.masks: 
            source = copy.deepcopy(self.audio_signal)
            source = source.apply_mask(mask)
            source.stft_params = self.stft_params
            source.istft(overwrite=True, truncate_to_length=self.audio_signal.signal_length)
            self.sources.append(source)
        # self.sources[0] -> harmonic, self.sources[1] -> percussive
        return self.sources
