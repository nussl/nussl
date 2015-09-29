import numpy as np
from scipy.io.wavfile import read,write
from WindowType import WindowType
import Constants
import f_istft, f_stft

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
    
    def __init__(self, inputFileName = None, timeSeries = None, signalStartingPosition = 0, signalLength = 0,
                 sampleRate = Constants.DEFAULT_SAMPLERATE):
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
        self.Time = np.mat([])
        self.SignalLength = signalLength
        self.nChannels = 0

        if (inputFileName is None) != (timeSeries is None): # XOR them
            pass

        if inputFileName is not None:
            self.LoadAudioFromFile(self.FileName, self.SignalLength, signalStartingPosition)
        elif timeSeries is not None:
            self.LoadAudioFromArray(timeSeries, sampleRate)
        
              
        # STFT properties
        self.ComplexSpectrogramData = np.mat([]) # complex spectrogram
        self.PowerSpectrumData = np.mat([]) # power spectrogram
        self.Fvec = np.mat([]) # freq. vector
        self.Tvec = np.mat([]) # time vector
        self.windowType = WindowType.DEFAULT
        self.windowLength = int(0.06*self.SampleRate)
        self.nfft = self.windowLength
        self.overlapRatio = 0.5
        self.overlapSamp = int(np.ceil(self.overlapRatio*self.windowLength))
        self.shouldMakePlot = False
        self.FrequencyMaxPlot = self.SampleRate/2
        
        #self.STFT() # is the spectrogram is required to be computed at start up
                
   
    
    def LoadAudioFromFile(self, inputFileName = None, signalStartingPosition = 0, signalLength = 0):
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
            self.SampleRate, audioInput = read(inputFileName)
        except Exception, e:
            print "Cannot read from file, {file}".format(file = inputFileName)
            raise e

        # scale to -1.0 to 1.0
        audioInput /= float(2**15) + 1.0       
        
        if audioInput.ndim < 2:
            audioInput = np.expand_dims(audioInput,axis=1)

        self.nChannels = np.shape(audioInput)[1]

        # the logic here needs work
        if signalLength == 0:
           self.AudioData = np.mat(audioInput) # make sure the signal is of matrix format
           self.SignalLength = np.shape(audioInput)[0]           
        else:
           self.SignalLength = int(signalLength * self.SampleRate)
           startPos = int(signalStartingPosition * self.SampleRate)
           self.AudioData = np.mat( audioInput[startPos: startPos+self.SignalLength, : ] )
        
        self.Time = np.mat( (1./self.SampleRate) * np.arange(self.SignalLength) ) 
        
   
    def LoadAudioFromArray(self, signal, sampleRate = Constants.DEFAULT_SAMPLERATE):
        """
        Loads an audio signal in numpy matrix format along with the sampling frequency

        """
        self.FileName = None
        self.AudioData = np.mat(signal) # each column contains one channel mixture
        self.SampleRate = sampleRate
        self.SignalLength, self.nChannels = np.shape(self.x)
        self.time = np.mat( (1./self.SampleRate) * np.arange(self.SignalLength) )

    
    def WriteAudioFile(self, outputFileName, sampleRate = Constants.DEFAULT_SAMPLERATE):
        """
        records the audio signal in a .wav file
        """
        if self.AudioData is None:
            raise Exception("Cannot write audio file because there is no audio data.")

        try:
            write(outputFileName, sampleRate, self.AudioData)
        except Exception, e:
            print "Cannot write to file, {file}.".format(file = outputFileName)
            raise e
        else:
            print "Successfully wrote {file}.".format(file = outputFileName)


    
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
            Xtemp, Ptemp, Ftemp, Ttemp = f_stft(self.x[:,i].T, self.windowlength, self.windowtype, \
                                                self.overlapSamp, self.SampleRate, nfft=self.nfft, mkplot=0)
                       
            if np.size(self.ComplexSpectrogramData) == 0:
                self.ComplexSpectrogramData = Xtemp
                self.PowerSpectrumData = Ptemp
                self.Fvec = Ftemp
                self.Tvec = Ttemp
            else:
                self.ComplexSpectrogramData = np.dstack([self.ComplexSpectrogramData, Xtemp]) 
                self.PowerSpectrumData = np.dstack([self.PowerSpectrumData, Ptemp])
        
        if self.shouldMakePlot:
            f_stft(self.x[:,0].T, self.windowlength, self.windowtype,
                       np.ceil(self.overlapRatio*self.windowlength), self.SampleRate, nfft=self.nfft,
                       mkplot=self.shouldMakePlot, fmax=self.fmaxplot)
            #plt.show()
            
        return self.ComplexSpectrogramData, self.PowerSpectrumData, self.Fvec, self.Tvec
            
    
    def iSTFT(self):
        """
        computes the inverse STFT and returns:
        self.x: time-domain signal       
        self.time: time vector 
        """
        if self.AudioData is None:
            raise e
         
        if np.size(self.ComplexSpectrogramData) == 0: 
            print("Empty spectrogrm matrix!")
            self.x = np.mat([])
            self.time = np.mat([])
        else:          
            self.x = np.mat([])
            for i in range(0, self.nChannels):
                x_temp, t_temp = f_istft(self.ComplexSpectrogramData[:,:,i], self.windowlength, self.windowtype,
                                         self.overlapSamp, self.SampleRate)
             
                if np.size(self.x) == 0:
                    self.x = np.mat(x_temp).T
                    self.time = np.mat(t_temp).T
                else:
                    self.x = np.hstack([self.x, np.mat(x_temp).T])
                     
            return self.x, self.time

    class __sample:
        def __init__(self):
            pass









