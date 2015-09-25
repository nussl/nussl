import numpy as np
import WindowType, WindowAttributes
import Constants
import AudioSignal

class SeparationBase:
	'''Base class for separation algorithms'''

	def __init__(self):
		self.SampleRate = Constants.DEFAULT_SAMPLERATE
		self.WindowAttributes = WindowAttributes(SampleRate)
		self.Mixture = AudioSignal()

	def plot();
		pass

	
