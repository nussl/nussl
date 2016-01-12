from __future__ import division

import numpy as np
import os.path
import scipy.io.wavfile as wav

from WindowType import WindowType
import FftUtils
import Constants


class AudioSignal(object):
    """
    The class Signal defines the properties of the audio signal object and performs
    basic operations such as Wav loading and computing the STFT/iSTFT.
    
    Read/write signal properties:
    - x: signal
    - signalLength: signal length (in number of samples)
    
    Read/write stft properties:
    - windowtype (e.g. 'Rectangular', 'Hamming', 'Hanning', 'Blackman')
    - windowlength (ms)
    - nfft (number of samples)
    - overlapRatio (in [0,1])
    - X: stft of the data
        
    Read-only properties:
    - SampleRate: sampling frequency
    - enc: encoding of the audio file
    - numCh: number of channels
  
    EXAMPLES:
    -create a new signal object:     sig=Signal('sample_audio_file.wav')  
    -compute the spectrogram of the new signal object:   sigSpec,sigPow,F,T=sig.STFT()
    -compute the inverse stft of a spectrogram:          sigrec,tvec=sig.iSTFT()
  
    """

    def __init__(self, pathToInputFile=None, timeSeries=None, signalStartingPosition=0, signalLength=0,
                 sampleRate=Constants.DEFAULT_SAMPLE_RATE, stft=None):
        """
        inputs: 
        pathToInputFile is a string indicating a path to a .wav file
        signalLength (in seconds): optional input indicating the length of the signal to be extracted. 
                             Default: full length of the signal
        signalStartingPosition (in seconds): optional input indicating the starting point of the section to be 
                               extracted. Default: 0 seconds
        timeSeries: Numpy matrix containing a time series
        SampleRate: sampling rate                       
        """

        self.PathToInputFile = pathToInputFile
        self._audioData = None
        self.time = np.array([])
        #self.SignalLength = signalLength
        #self.nChannels = 1
        self.SampleRate = sampleRate

        if (pathToInputFile is not None) and (timeSeries is not None):
            raise Exception('Cannot initialize AudioSignal object with a path AND an array!')

        if pathToInputFile is not None:
            self.LoadAudioFromFile(self.PathToInputFile, signalLength, signalStartingPosition)
        elif timeSeries is not None:
            self.LoadAudioFromArray(timeSeries, sampleRate)

        # STFT properties
        self.ComplexSpectrogramData = np.array([]) if stft is None else stft  # complex spectrogram
        self.PowerSpectrumData = np.array([])  # power spectrogram
        self.Fvec = np.array([])  # freq. vector
        self.Tvec = np.array([])  # time vector

        # TODO: put these in a WindowAttributes object and wrap in a property
        self.windowType = WindowType.DEFAULT
        self.windowLength = int(0.06 * self.SampleRate)
        self.nfft = self.windowLength
        self.overlapRatio = 0.5
        self.overlapSamp = int(np.ceil(self.overlapRatio * self.windowLength))
        self.FrequencyMaxPlot = self.SampleRate / 2

    ##################################################
    # Properties
    ##################################################

    # Constants for accessing _audioData np.array indices
    LEN = 1
    CHAN = 0

    @property
    def SignalLength(self):
        return self._audioData.shape[self.LEN]

    @property
    def nChannels(self):
        return self._audioData.shape[self.CHAN]

    @property
    def AudioData(self):
        return self._audioData

    @AudioData.setter
    def AudioData(self, value):
        assert (type(value) == np.ndarray)

        self._audioData = value

        if self._audioData.ndim < 2:
            self._audioData = np.expand_dims(self._audioData, axis=self.CHAN)

    @property
    def FileName(self):
        if self.PathToInputFile is not None:
            return os.path.split(self.PathToInputFile)[1]
        return None


    ##################################################
    # I/O
    ##################################################

    def LoadAudioFromFile(self, inputFileName, signalStartingPosition=0, signalLength=0):
        """
        Loads an audio signal from a .wav file
        signalLength (in seconds): optional input indicating the length of the signal to be extracted. 
            signalLength of 0 means read the whole file
            Default: full length of the signal
        signalStartingPosition (in seconds): optional input indicating the starting point of the section to be 
            extracted.
            Default: 0 seconds
        
        """
        try:
            self.SampleRate, audioInput = wav.read(inputFileName)
        except Exception, e:
            print "Cannot read from file, {file}".format(file=inputFileName)
            raise e

        audioInput = audioInput.T

        # Change from fixed point to floating point
        audioInput = audioInput.astype('float') / (np.iinfo(audioInput.dtype).max + 1.0)

        # TODO: the logic here needs work
        if signalLength == 0:
            self.AudioData = np.array(audioInput)
        else:
            startPos = int(signalStartingPosition * self.SampleRate)
            self.AudioData = np.array(audioInput[startPos: startPos + self.SignalLength, :])

        self.time = np.array((1. / self.SampleRate) * np.arange(self.SignalLength))

    def LoadAudioFromArray(self, signal, sampleRate=Constants.DEFAULT_SAMPLE_RATE):
        """
        Loads an audio signal in numpy matrix format along with the sampling rate

        """
        self.PathToInputFile = None
        self.AudioData = np.array(signal)
        self.SampleRate = sampleRate

        self.time = np.array((1. / self.SampleRate) * np.arange(self.SignalLength))

    # TODO: verbose toggle
    def WriteAudioFile(self, outputFileName, sampleRate=None, verbose=False):
        """
        records the audio signal in a .wav file
        """
        if self.AudioData is None:
            raise Exception("Cannot write audio file because there is no audio data.")

        try:
            self.PeakNormalize()

            if sampleRate is None:
                sampleRate = self.SampleRate

            wav.write(outputFileName, sampleRate, self.AudioData.T)
        except Exception, e:
            print "Cannot write to file, {file}.".format(file=outputFileName)
            raise e
        if verbose:
            print "Successfully wrote {file}.".format(file=outputFileName)

    ##################################################
    #               STFT Utilities
    ##################################################

    def STFT(self):
        """
        computes the STFT of the audio signal
        :returns
            self.ComplexSpectrogramData: complex stft
            self.PowerSpectrumData: power spectrogram
            self.Fvec: frequency vector
            self.Tvec: vector of time frames
        """
        if self.AudioData is None:
            raise Exception("No audio data to make STFT from.")

        for i in range(1, self.nChannels + 1):
            Xtemp, Ptemp, Ftemp, Ttemp = FftUtils.f_stft(self.getChannel(i).T, nFfts=self.nfft,
                                                         winLength=self.windowLength, windowType=self.windowType,
                                                         winOverlap=self.overlapSamp, sampleRate=self.SampleRate,
                                                         mkplot=0)

            if np.size(self.ComplexSpectrogramData) == 0:
                self.ComplexSpectrogramData = Xtemp
                self.PowerSpectrumData = Ptemp
                self.Fvec = Ftemp
                self.Tvec = Ttemp
            else:
                self.ComplexSpectrogramData = np.dstack([self.ComplexSpectrogramData, Xtemp])
                self.PowerSpectrumData = np.dstack([self.PowerSpectrumData, Ptemp])

        return self.ComplexSpectrogramData, self.PowerSpectrumData, self.Fvec, self.Tvec

    def iSTFT(self):
        """
        Computes and returns the inverse STFT.
        Will overwrite any data in self.AudioData!

        :returns: self.AudioData: time-domain signal and self.time: time vector
        """
        if self.ComplexSpectrogramData.size == 0:
            raise Exception('Cannot do inverse STFT without STFT data!')

        self.AudioData = np.array([])
        for i in range(1, self.nChannels + 1):
            x_temp, t_temp = FftUtils.f_istft(self.ComplexSpectrogramData, self.windowLength,
                                              self.windowType,
                                              self.overlapSamp, self.SampleRate)

            if np.size(self.AudioData) == 0:
                self.AudioData = np.array(x_temp).T
                self.time = np.array(t_temp).T
            else:
                self.AudioData = np.hstack([self.AudioData, np.array(x_temp).T])

        if len(self.AudioData.shape) == 1:
            self.AudioData = np.expand_dims(self.AudioData, axis=1)

        return self.AudioData, self.time

    ##################################################
    #                  Utilities
    ##################################################

    def concat(self, other):
        """

        :param other:
        :return:
        """
        if self.nChannels != other.nChannels:
            raise Exception('Cannot concat two signals that have a different number of channels!')

        self.AudioData = np.concatenate((self.AudioData, other.AudioData))

    def getChannel(self, n):
        """
        Gets the n-th channel. 1-based.
        :param n: index of channel to get 1-based.
        :return: n-th channel.
        """
        return self.AudioData[n - 1,]

    def PeakNormalize(self, bitDepth=16):
        """
        Normalizes the whole audio file to the max value.
        :param bitDepth: Max value (in bits) to normalize to. Default is 16 bits.
        :return:
        """
        bitDepth -= 1
        maxVal = 1.0
        maxSignal = np.max(np.abs(self.AudioData))
        if maxSignal > maxVal:
            self.AudioData = np.divide(self.AudioData, maxSignal)

    ##################################################
    #              Operator overloading
    ##################################################

    def __add__(self, other):
        if self.nChannels != other.nChannels:
            raise Exception('Cannot add two signals that have a different number of channels!')

        # for ch in range(self.nChannels):
        # TODO: make this work for multiple channels
        if self.AudioData.size > other.AudioData.size:
            combined = np.copy(self.AudioData)
            combined[0: other.AudioData.size] += other.AudioData
        else:
            combined = np.copy(other.AudioData)
            combined[0: self.AudioData.size] += self.AudioData

        return AudioSignal(timeSeries=combined)

    def __sub__(self, other):
        if self.nChannels != other.nChannels:
            raise Exception('Cannot subtract two signals that have a different number of channels!')

        # for ch in range(self.nChannels):
        # TODO: make this work for multiple channels
        if self.AudioData.size > other.AudioData.size:
            combined = np.copy(self.AudioData)
            combined[0: other.AudioData.size] -= other.AudioData
        else:
            combined = np.copy(other.AudioData)
            combined[0: self.AudioData.size] -= self.AudioData

        return AudioSignal(timeSeries=combined)

    def __iadd__(self, other):
        return self + other

    def __isub__(self, other):
        return self - other

    def __len__(self):
        return len(self.AudioData)

