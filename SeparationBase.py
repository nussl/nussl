import numpy as np
import WindowType, WindowAttributes
import Constants
import AudioSignal

class SeparationBase:
	'''Base class for separation algorithms'''

	def __init__(self, windowAttributes = None, sampleRate = None, audioSignal = None):

		if sampleRate is not None:
			self.SampleRate = sampleRate
		else:
			self.SampleRate = Constants.DEFAULT_SAMPLERATE

		if windowAttributes is not None:
			self.WindowAttributes = audioSignal
		else:
			self.WindowAttributes = WindowAttributes(SampleRate)

		if audioSignal is not None:
			self.Mixture = audioSignal
		else:
			self.Mixture = AudioSignal()

	def Plot():
		pass

	def Run():
		pass
