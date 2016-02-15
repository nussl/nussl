#!/usr/bin/env python
# -*- coding: utf-8 -*-


class WindowType:
    RECTANGULAR = 'Rectangular'
    HAMMING = 'Hamming'
    HANN = 'Hann'
    BLACKMAN = 'Blackman'
    DEFAULT = HAMMING

    def __init__(self):
        # TODO: delete this
        raise DeprecationWarning('Moved to spectral_utils.py')
