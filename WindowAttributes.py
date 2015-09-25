import WindowType, Constants
import numpy as np

class WindowAttributes:
	""" """

	def __init__(self, sampleRate):
		self.WindowType = WindowType.DEFAULT
		self.WindowLength = int(2**(np.ceil(np.log2(Constants.DEFAULT_WIN_LEN_PARAM*sampleRate))))
		self.WindowType = WindowType.DEFAULT
		self.WindowOverlap = WindowLength / 2
