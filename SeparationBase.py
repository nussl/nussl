import WindowAttributes
import Constants
import AudioSignal


class SeparationBase(object):
    '''Base class for separation algorithms'''

    def __init__(self, windowAttributes=None, sampleRate=None, audioSignal=None):

        if sampleRate is not None:
            self.SampleRate = sampleRate
        else:
            self.SampleRate = Constants.DEFAULT_SAMPLERATE

        if windowAttributes is not None:
            self.WindowAttributes = windowAttributes
        else:
            self.WindowAttributes = WindowAttributes.WindowAttributes(self.SampleRate)

        if audioSignal is not None:
            self.Mixture = audioSignal
        else:
            self.Mixture = AudioSignal.AudioSignal()

    def Plot(self):
        pass

    def Run(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        self.Run(*args, **kwargs)
