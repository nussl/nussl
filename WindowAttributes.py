import Constants
import numpy as np
from WindowType import WindowType


class WindowAttributes:
    """ """

    def __init__(self, sampleRate):
        self.WindowLength = int(2 ** (np.ceil(np.log2(Constants.DEFAULT_WIN_LEN_PARAM * sampleRate))))
        self.WindowType = WindowType.DEFAULT
        self.WindowOverlap = self.WindowLength / 2
        self.Nfft = self.WindowLength
