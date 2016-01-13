"""
Base class for all separation algorithms.
Do not call this. It will not do anything.

Authors: Fatameh Pishdadian and Ethan Manilow
Interactive Audio Lab
Northwestern University, 2015

"""

import WindowAttributes
import Constants
import AudioSignal


class SeparationBase(object):
    """
    Base class for all separation algorithms
    """

    def __init__(self, windowAttributes=None, sampleRate=None, audioSignal=None):

        if sampleRate is not None:
            self.SampleRate = sampleRate
        else:
            self.SampleRate = Constants.DEFAULT_SAMPLE_RATE

        if windowAttributes is not None:
            self.WindowAttributes = windowAttributes
        else:
            self.WindowAttributes = WindowAttributes.WindowAttributes(self.SampleRate)

        if audioSignal is not None:
            self.Mixture = audioSignal
        else:
            self.Mixture = AudioSignal.AudioSignal()

    def Plot(self, outputName, **kwargs):
        """
        Plots relevant data for separation algorithm
        :return:
        """
        raise NotImplementedError('Cannot call base class.')

    def Run(self):
        """
        Run separation algorithm
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError('Cannot call base class.')

    def MakeAudioSignals(self):
        """
        Makes AudioSignal objects after separation algorithm is run
        :return:
        """
        raise NotImplementedError('Cannot call base class.')

    def __call__(self):
        self.Run()
