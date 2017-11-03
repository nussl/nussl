#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for verifying lists of objects. Used in many separation classes that accept a list of AudioSignals,
or in evaluation classes that accept lists of SeparationBase objects.
"""

__all__ = ['audio_signal_list_lax', 'audio_signal_list_strict', 'separation_base_list', 'mask_separation_base_list']


def audio_signal_list_lax(audio_signal_list):
    """
    Verifies that an input (audio_signal_list) is a list of :ref:`AudioSignal` objects. If not so, attempts
    to correct the list (if possible) and returns the corrected list.
    Args:
        audio_signal_list (list): List of :ref:`AudioSignal` objects

    Returns:
        audio_signal_list (list): Verified list of :ref:`AudioSignal` objects.

    """
    # Lazy load to prevent a circular reference at load time
    from .audio_signal import AudioSignal

    if isinstance(audio_signal_list, AudioSignal):
        audio_signal_list = [audio_signal_list]
    elif isinstance(audio_signal_list, list):
        if not all(isinstance(s, AudioSignal) for s in audio_signal_list):
            raise ValueError('All input objects must be AudioSignal objects!')
        if not all(s.has_data for s in audio_signal_list):
            raise ValueError('All AudioSignal objects in input list must have data!')
    else:
        raise ValueError('All input objects must be AudioSignal objects!')

    return audio_signal_list


def audio_signal_list_strict(audio_signal_list):
    """
    Verifies that an input (audio_signal_list) is a list of :ref:`AudioSignal` objects and that they all have the same
    sample rate and same number of channels. If not true, attempts to correct the list (if possible) and returns
    the corrected list.

    Args:
        audio_signal_list (list): List of :ref:`AudioSignal` objects

    Returns:
        audio_signal_list (list): Verified list of :ref:`AudioSignal` objects, that all have
        the same sample rate and number of channels.

    """
    audio_signal_list = audio_signal_list_lax(audio_signal_list)

    if not all(audio_signal_list[0].sample_rate == s.sample_rate for s in audio_signal_list):
        raise ValueError('All input AudioSignal objects must have the same sample rate!')

    if not all(audio_signal_list[0].num_channels == s.num_channels for s in audio_signal_list):
        raise ValueError('All input AudioSignal objects must have the same number of channels!')

    return audio_signal_list


def separation_base_list(separation_list):
    """
    Verifies that all items in `separation_list` are :ref:`SeparationBase` -derived objects. If not so, attempts
    to correct the list if possible and returns the corrected list.

    Args:
        separation_list: (list) List of :ref:`SeparationBase` -derived objects

    Returns:
        separation_list: (list) Verified list of :ref:`SeparationBase` -derived objects

    """
    # Lazy load to prevent a circular reference at load time
    from ..separation import SeparationBase

    if isinstance(separation_list, SeparationBase):
        separation_list = [separation_list]
    elif isinstance(separation_list, list):
        if not all(isinstance(s, SeparationBase) for s in separation_list):
            raise ValueError('All separation objects must be SeparationBase-derived objects!')
    else:
        raise ValueError('All separation objects must be SeparationBase-derived objects!')

    return separation_list


def mask_separation_base_list(mask_separation_list):
    """
    Verifies that all items in `separation_list` are :ref:`MaskSeparationBase` -derived objects. If not so, attempts
    to correct the list if possible and returns the corrected list.

    Args:
        mask_separation_list: (list) List of :ref:`MaskSeparationBase` -derived objects

    Returns:
        separation_list: (list) Verified list of :ref:`MaskSeparationBase` -derived objects

    """
    # Lazy load to prevent a circular reference at load time
    from ..separation import MaskSeparationBase

    if isinstance(mask_separation_list, MaskSeparationBase):
        mask_separation_list = [mask_separation_list]
    elif isinstance(mask_separation_list, list):
        if not all(isinstance(s, MaskSeparationBase) for s in mask_separation_list):
            raise ValueError('All separation objects must be MaskSeparationBase-derived objects!')
    else:
        raise ValueError('All separation objects must be MaskSeparationBase-derived objects!')

    return mask_separation_list
