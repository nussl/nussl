from __future__ import division
import numpy as np
import scipy.io.wavfile as wav
from WindowType import WindowType
import f_istft, FftUtils, Constants


class AudioSignal:
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

    def __init__(self, inputFileName=None, timeSeries=None, signalStartingPosition=0, signalLength=0,
                 sampleRate=Constants.DEFAULT_SAMPLERATE, stft=None):
        """
        inputs: 
        inputFileName is a string indicating a path to a .wav file
        signalLength (in seconds): optional input indicating the length of the signal to be extracted. 
                             Default: full length of the signal
        signalStartingPosition (in seconds): optional input indicating the starting point of the section to be 
                               extracted. Default: 0 seconds
        timeSeries: Numpy matrix containing a time series
        SampleRate: sampling rate                       
        """

        self.FileName = inputFileName
        self.AudioData = None
        self.Time = np.array([])
        self.SignalLength = signalLength
        self.nChannels = 1
        self.SampleRate = sampleRate

        if (inputFileName is None) != (timeSeries is None):  # XOR them
            pass

        if inputFileName is not None:
            self.LoadAudioFromFile(self.FileName, self.SignalLength, signalStartingPosition)
        elif timeSeries is not None:
            self.LoadAudioFromArray(timeSeries, sampleRate)

        # STFT properties
        self.ComplexSpectrogramData = np.array([]) if stft is None else stft  # complex spectrogram
        self.PowerSpectrumData = np.array([])  # power spectrogram
        self.Fvec = np.array([])  # freq. vector
        self.Tvec = np.array([])  # time vector
        self.windowType = WindowType.DEFAULT
        self.windowLength = int(0.06 * self.SampleRate)
        self.nfft = self.windowLength
        self.overlapRatio = 0.5
        self.overlapSamp = int(np.ceil(self.overlapRatio * self.windowLength))
        self.shouldMakePlot = False
        self.FrequencyMaxPlot = self.SampleRate / 2

        # self.STFT() # is the spectrogram is required to be computed at start up

    def LoadAudioFromFile(self, inputFileName=None, signalStartingPosition=0, signalLength=0):
        """
        Loads an audio signal from a .wav file
        signalLength (in seconds): optional input indicating the length of the signal to be extracted. 
            signalLength of 0 means read the whole file
            Default: full length of the signal
        signalStartingPosition (in seconds): optional input indicating the starting point of the section to be 
            extracted.
            Default: 0 seconds
        
        """
        if inputFileName is None:
            raise Exception("Cannot load audio from file because there is no file to open from!")

        try:
            self.SampleRate, audioInput = wav.read(inputFileName)
        except Exception, e:
            print "Cannot read from file, {file}".format(file=inputFileName)
            raise e

        # scale to -1.0 to 1.0
        np.divide(audioInput, float(2 ** 15) + 1.0)

        if audioInput.ndim < 2:
            audioInput = np.expand_dims(audioInput, axis=1)

        self.nChannels = np.shape(audioInput)[1]

        # the logic here needs work
        if signalLength == 0:
            self.AudioData = np.array(audioInput)  # make sure the signal is of matrix format
            self.SignalLength = np.shape(audioInput)[0]
        else:
            self.SignalLength = int(signalLength * self.SampleRate)
            startPos = int(signalStartingPosition * self.SampleRate)
            self.AudioData = np.array(audioInput[startPos: startPos + self.SignalLength, :])

        self.Time = np.array((1. / self.SampleRate) * np.arange(self.SignalLength))

    def NormalizeSingal(self):
        pass

    def LoadAudioFromArray(self, signal, sampleRate=Constants.DEFAULT_SAMPLERATE):
        """
        Loads an audio signal in numpy matrix format along with the sampling frequency

        """
        self.FileName = None
        self.AudioData = np.array(signal)  # each column contains one channel mixture
        self.SampleRate = sampleRate
        if len(self.AudioData.shape) > 1:
            self.SignalLength, self.nChannels = np.shape(self.AudioData)
        else:
            self.SignalLength, = self.AudioData.shape
            self.nChannels = 1
        self.time = np.array((1. / self.SampleRate) * np.arange(self.SignalLength))

    # TODO: verbose toggle
    def WriteAudioFile(self, outputFileName, sampleRate=Constants.DEFAULT_SAMPLERATE):
        """
        records the audio signal in a .wav file
        """
        if self.AudioData is None:
            raise Exception("Cannot write audio file because there is no audio data.")

        try:
            wav.write(outputFileName, sampleRate, self.AudioData)
        except Exception, e:
            print "Cannot write to file, {file}.".format(file=outputFileName)
            raise e
        else:
            print "Successfully wrote {file}.".format(file=outputFileName)

    def STFT(self):
        """
        computes the STFT of the audio signal and returns:
        self.ComplexSpectrogramData: complex stft
        self.PowerSpectrumData: power spectrogram
        self.Fvec: frequency vector
        self.Tvec: vector of time frames
        """
        if self.AudioData is None:
            raise Exception("No audio data to make STFT from.")

        for i in range(0, self.nChannels):
            Xtemp, Ptemp, Ftemp, Ttemp = FftUtils.f_stft(self.AudioData[:, i].T, self.windowLength, self.windowType,
                                                         self.overlapSamp, self.SampleRate, nFfts=self.nfft, mkplot=0)

            if np.size(self.ComplexSpectrogramData) == 0:
                self.ComplexSpectrogramData = Xtemp
                self.PowerSpectrumData = Ptemp
                self.Fvec = Ftemp
                self.Tvec = Ttemp
            else:
                self.ComplexSpectrogramData = np.dstack([self.ComplexSpectrogramData, Xtemp])
                self.PowerSpectrumData = np.dstack([self.PowerSpectrumData, Ptemp])

        if self.shouldMakePlot:
            FftUtils.f_stft(self.AudioData[:, 0].T, self.windowLength, self.windowType,
                            np.ceil(self.overlapRatio * self.windowLength), self.SampleRate, nFfts=self.nfft,
                            mkplot=self.shouldMakePlot, fmax=self.FrequencyMaxPlot)
            # plt.show()

        return self.ComplexSpectrogramData, self.PowerSpectrumData, self.Fvec, self.Tvec

    def iSTFT(self):
        """
        computes the inverse STFT and returns:
        self.AudioData: time-domain signal       
        self.time: time vector 
        """
        if self.ComplexSpectrogramData.size == 0:
            raise Exception('Cannot do inverse STFT without STFT data!')

        self.AudioData = np.array([])
        for i in range(0, self.nChannels):
            x_temp, t_temp = FftUtils.f_istft(self.ComplexSpectrogramData, self.windowLength,
                                              self.windowType,
                                              self.overlapSamp, self.SampleRate)

            if np.size(self.AudioData) == 0:
                self.AudioData = np.array(x_temp).T
                self.time = np.array(t_temp).T
            else:
                self.AudioData = np.hstack([self.AudioData, np.array(x_temp).T])

        return self.AudioData, self.time

    # TODO: Some kinks to work out here
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

    # TODO: += operator

    class __sample:
        def __init__(self):
            pass
