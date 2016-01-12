import Constants
import numpy as np
from WindowType import WindowType


class WindowAttributes(object):
    """
    The WindowAttributes class is a container for information regarding windowing.
    This object will get passed around instead of each of these individual attributes.
    """

    def __init__(self, sampleRate, windowLength=None, windowType=None, windowOverlap=None, nFft=None):
        defaultWinLen = int(2 ** (np.ceil(np.log2(Constants.DEFAULT_WIN_LEN_PARAM * sampleRate))))
        self._windowLength = defaultWinLen if windowLength is None else windowLength
        self.WindowType = WindowType.DEFAULT if windowType is None else windowType
        self._windowOverlap = self.WindowLength / 2 if windowOverlap is None else windowOverlap
        self._nfft = self.WindowLength if nFft is None else nFft

        if windowOverlap is None:
            self._windowOverlapNeedsUpdate = True
        if nFft is None:
            self._nfftNeedsUpdate = True

    @property
    def WindowLength(self):
        return self._windowLength

    @WindowLength.setter
    def WindowLength(self, value):
        """
        Length of window in samples. If WindowOverlap or Nfft are not set manually,
        then changing this will update them to WindowOverlap = WindowLength / 2, and
        and Nfft = WindowLength
        :param value:
        :return:
        """
        self._windowLength = value

        if self._windowOverlapNeedsUpdate:
            self._windowOverlap = value / 2

        if self._nfftNeedsUpdate:
            self._nfft = value

    @property
    def WindowOverlap(self):
        return self._windowOverlap

    @WindowOverlap.setter
    def WindowOverlap(self, value):
        """
        Overlap of windows.
        By default this is linked to WindowLength (value of WindowLength / 2),
        but if this is set manually then they are both independent.
        :param value:
        :return:
        """
        self._windowOverlapNeedsUpdate = False
        self._windowOverlap = value

    @property
    def Nfft(self):
        return self._nfft

    @Nfft.setter
    def Nfft(self, value):
        """
        Number of FFT bins.
        By default this is linked to WindowLength (value of WindowLength),
        but if this is set manually then they are both independent.
        :param value:
        :return:
        """
        self._nfftNeedsUpdate = False
        self._nfft = value
