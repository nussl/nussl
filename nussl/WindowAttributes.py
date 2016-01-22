#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

import Constants
from WindowType import WindowType


class WindowAttributes(object):
    """
    The window_attributes class is a container for information regarding windowing.
    This object will get passed around instead of each of these individual attributes.
    """

    def __init__(self, sample_rate, window_length=None, window_type=None, window_overlap=None, num_fft=None):
        default_win_len = int(2 ** (np.ceil(np.log2(Constants.DEFAULT_WIN_LEN_PARAM * sample_rate))))
        self._window_length = default_win_len if window_length is None else window_length
        self.window_type = WindowType.DEFAULT if window_type is None else window_type
        self._window_overlap = self.window_length / 2 if window_overlap is None else window_overlap
        self._num_fft = self.window_length if num_fft is None else num_fft

        if window_overlap is None:
            self._window_overlap_needs_update = True
        if num_fft is None:
            self._num_fft_needs_update = True

    @property
    def window_length(self):
        return self._window_length

    @window_length.setter
    def window_length(self, value):
        """
        Length of window in samples. If window_overlap or num_fft are not set manually,
        then changing this will update them to window_overlap = window_length / 2, and
        and num_fft = window_length
        :param value:
        :return:
        """
        self._window_length = value

        if self._window_overlap_needs_update:
            self._window_overlap = value / 2

        if self._num_fft_needs_update:
            self._num_fft = value

    @property
    def window_overlap(self):
        return self._window_overlap

    @window_overlap.setter
    def window_overlap(self, value):
        """
        Overlap of windows.
        By default this is linked to window_length (value of window_length / 2),
        but if this is set manually then they are both independent.
        :param value:
        :return:
        """
        self._window_overlap_needs_update = False
        self._window_overlap = value

    @property
    def num_fft(self):
        return self._num_fft

    @num_fft.setter
    def num_fft(self, value):
        """
        Number of FFT bins.
        By default this is linked to window_length (value of window_length),
        but if this is set manually then they are both independent.
        :param value:
        :return:
        """
        self._num_fft_needs_update = False
        self._num_fft = value
