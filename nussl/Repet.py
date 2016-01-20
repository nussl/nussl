"""
Implements the REpeating Pattern Extraction Technique algorithm using the Similarity Matrix (REPET-SIM). REPET is a
simple method for separating the repeating background from the non-repeating foreground in a piece of audio mixture.
REPET-SIM is a generalization of REPET, which looks for similarities instead of periodicities.

References:

    * Zafar Rafii and Bryan Pardo. "Audio Separation System and Method," US20130064379 A1, US 13/612,413, March 14, 2013
    * Zafar Rafii and Bryan Pardo. "Music/Voice Separation using the Similarity Matrix," 13th International Society on
      Music Information Retrieval, Porto, Portugal, October 8-12, 2012.

See Also:
    http://music.eecs.northwestern.edu/research.php?project=repet

"""

import numpy as np
import scipy.fftpack as scifft
import FftUtils
import SeparationBase
import Constants
import AudioSignal


class Repet(SeparationBase.SeparationBase):
    """
        Parameters:
            audioSignal (AudioSignal): audio mixture (M by N) containing M channels and N time samples
            Type (RepetType): Variant of Repet algorithm to perform.
            windowAttributes (WindowAttributes): WindowAttributes object describing the window used in the repet
             algorithm
            sampleRate (int): the sample rate of the audio signal
            HighPassCutoff (Optional[int]): Defaults to 100
            similarityThreshold (Optional[int]): Used for RepetType.SIM. Defaults to 0
            MinDistanceBetweenFrames (Optional[int]): Used for RepetType.SIM. Defaults to 1
            MaxRepeatingFrames (Optional[int]): Used for RepetType.SIM. Defaults to 10
            Period (Optional[float]): Used for RepetType.ORIGINAL. The Period of the repeating part of the signal.
            MinPeriod (Optional[float]): Used for RepetType.ORIGINAL. Only used if Period is not provided. Defaults to
             0.8
            MaxPeriod (Optional[float]): Used for RepetType.ORIGINAL. Only used if Period is not provided. Defaults to
             min(8, self.Mixture.SignalLength/3)

        """

    def __init__(self, audioSignal, Type=None, windowAttributes=None, sampleRate=None,
                 similarityThreshold=None, MinDistanceBetweenFrames=None, MaxRepeatingFrames=None,
                 MinPeriod=None, MaxPeriod=None, Period=None, HighPassCutoff=None):

        self.__dict__.update(locals())
        super(Repet, self).__init__(windowAttributes=windowAttributes, sampleRate=sampleRate,
                                    audioSignal=audioSignal)
        self.type = RepetType.DEFAULT if Type is None else Type
        self.HighPassCutoff = 100 if HighPassCutoff is None else HighPassCutoff

        if self.type == RepetType.SIM:
            # if similarityThreshold == 0:
            #     raise Warning('Using default value of 1 for similarityThreshold.')
            # if MinDistanceBetweenFrames == 1:
            #     raise Warning('Using default value of 0 for MinDistanceBetweenFrames.')
            # if MaxRepeatingFrames == 100:
            #     raise Warning('Using default value of 100 for maxRepeating frames.')

            self.SimilarityThreshold = 0 if similarityThreshold is None else similarityThreshold
            self.MinDistanceBetweenFrames = 1 if MinDistanceBetweenFrames is None else MinDistanceBetweenFrames
            self.MaxRepeatingFrames = 10 if MaxRepeatingFrames is None else MaxRepeatingFrames
        elif self.type == RepetType.ORIGINAL:

            if Period is None:
                self.MinPeriod = 0.8 if MinPeriod is None else MinPeriod
                self.MaxPeriod = min(8, self.Mixture.SignalLength / 3) if MaxPeriod is None else MaxPeriod
                self.MinPeriod = self._updatePeriod(self.MinPeriod)
                self.MaxPeriod = self._updatePeriod(self.MaxPeriod)
            else:
                self.Period = Period
                self.Period = self._updatePeriod(self.Period)

        elif self.type == RepetType.ADAPTIVE:
            raise NotImplementedError('Not allowed to do this yet!')
        else:
            raise TypeError('\'Type\' in Repet() constructor cannot be {}'.format(Type))
        self.Verbose = False

    # @property
    def Run(self):
        """Runs the REPET algorithm

        Returns:
            y (AudioSignal): repeating background (M by N) containing M channels and N time samples
            (the corresponding non-repeating foreground is equal to x-y)

        EXAMPLE:
             ::
            signal = nussl.AudioSignal(pathToInputFile='inputName.wav')

            # Set up window parameters
            win = nussl.WindowAttributes(signal.SampleRate)
            win.WindowLength = 2048
            win.WindowType = nussl.WindowType.HAMMING

            # Set up and run Repet
            repet = nussl.Repet(signal, Type=nussl.RepetType.SIM, windowAttributes=win)
            repet.MinDistanceBetweenFrames = 0.1
            repet.Run()

        """

        # unpack window parameters
        win_len, win_type, win_ovp, nfft = self.WindowAttributes.WindowLength, self.WindowAttributes.WindowType, \
                                           self.WindowAttributes.WindowOverlap, self.WindowAttributes.Nfft

        # High pass filter cutoff freq. (in # of freq. bins)
        self.HighPassCutoff = np.ceil(float(self.HighPassCutoff) * (self.WindowAttributes.Nfft - 1) / self.SampleRate)

        self._computeSpectrum()

        # Run the specific algorithm
        mask = None
        S = None
        if self.type == RepetType.SIM:
            S = self._doRepetSim()
            mask = self.ComputeRepeatingMaskSim

        elif self.type == RepetType.ORIGINAL:
            S = self._doRepetOriginal()
            mask = self.ComputeRepeatingMaskBeat

        elif self.type == RepetType.ADAPTIVE:
            raise NotImplementedError('How did you get into this state????')

        # separate the mixture background by masking
        N, M = self.Mixture.AudioData.shape
        self.bkgd = np.zeros_like(self.Mixture.AudioData)
        for i in range(N):
            RepMask = mask(self.RealSpectrum[:, :, i], S)
            RepMask[1:self.HighPassCutoff, :] = 1  # high-pass filter the foreground
            XMi = RepMask * self.ComplexSpectrum[:, :, i]
            yi = FftUtils.f_istft(XMi, win_len, win_type, win_ovp, self.SampleRate)[0]
            self.bkgd[i,] = yi[0:M]

        # self.bkgd = self.bkgd.T
        self.bkgd = AudioSignal.AudioSignal(timeSeries=self.bkgd)

        return self.bkgd

    def _computeSpectrum(self):

        # compute the spectrograms of all channels
        N, M = self.Mixture.AudioData.shape
        self.ComplexSpectrum = FftUtils.f_stft(self.Mixture.getChannel(1), windowAttributes=self.WindowAttributes,
                                               sampleRate=self.SampleRate)[0]

        for i in range(1, N):
            Sx = FftUtils.f_stft(self.Mixture.getChannel(i), windowAttributes=self.WindowAttributes,
                                 sampleRate=self.SampleRate)[0]
            self.ComplexSpectrum = np.dstack([self.ComplexSpectrum, Sx])

        self.RealSpectrum = np.abs(self.ComplexSpectrum)
        if N == 1:
            self.ComplexSpectrum = self.ComplexSpectrum[:, :, np.newaxis]
            self.RealSpectrum = self.RealSpectrum[:, :, np.newaxis]

    def GetSimilarityMatrix(self):
        """Calculates and returns the similarity matrix for the audio file associated with this object

        Returns:
             V (np.array): similarity matrix for the audio file.

        """
        self._computeSpectrum()
        V = np.mean(self.RealSpectrum, axis=2)
        self.SimilarityMatrix = self.ComputeSimilarityMatrix(V)
        return self.SimilarityMatrix

    def GetBeatSpectrum(self):
        """Calculates and returns the beat spectrum for the audio file associated with this object

        Returns:
            B (np.array): beat spectrum for the audio file

        """
        self._computeSpectrum()
        self.BeatSpectrum = self.ComputeBeatSpectrum(np.mean(self.RealSpectrum ** 2, axis=2))
        return self.BeatSpectrum

    def _doRepetSim(self):
        # unpack window parameters
        len, type, ovp, nfft = self.WindowAttributes.WindowLength, self.WindowAttributes.WindowType, \
                               self.WindowAttributes.WindowOverlap, self.WindowAttributes.Nfft

        Vavg = np.mean(self.RealSpectrum, axis=2)
        S = self.ComputeSimilarityMatrix(Vavg)

        self.MinDistanceBetweenFrames = np.round(self.MinDistanceBetweenFrames * self.SampleRate / ovp)
        S = self.FindSimilarityIndices(S)

        return S

    def _doRepetOriginal(self):
        self.BeatSpectrum = self.ComputeBeatSpectrum(np.mean(self.RealSpectrum ** 2, axis=2))
        self.RepeatingPeriod = self.FindRepeatingPeriod(self.BeatSpectrum, self.MinPeriod, self.MaxPeriod)
        return self.RepeatingPeriod

    @staticmethod
    def ComputeSimilarityMatrix(X):
        """
        Computes the similarity matrix using the cosine similarity for input matrix X.
        
        Parameters:
            X (np.array): 2D matrix containing the magnitude spectrogram of the audio signal (Lf by Lt)
        Returns:
            S (np.array): similarity matrix (Lt by Lt)
        """
        assert (type(X) == np.ndarray)

        # normalize the columns of the magnitude spectrogram
        Lt = X.shape[1]
        X = X.T
        for i in range(0, Lt):
            Xi = X[i, :]
            rowNorm = np.sqrt(np.dot(Xi, Xi))
            X[i, :] = Xi / (rowNorm + Constants.EPSILON)

        # compute the similarity matrix    
        S = np.dot(X, X.T)
        return S

    def FindSimilarityIndices(self, S):
        """
        Finds the similarity indices for all time frames from the similarity matrix
        
        Parameters:
            S (np.array): similarity matrix (Lt by Lt)
            simparam (List): array containing 3 similarity parameters

                * simparam[0]: minimum threshold (in [0,1]) for the similarity measure within repeating frames
                * simparam[1]: minimum distance (in # of time frames) between repeating frames
                * simparam[2]: maximum number of repeating frames for the median filter
                 
        Returns:
            I (np.array): similarity indices for all time frames
        """

        Lt = S.shape[0]
        I = np.zeros((Lt, self.MaxRepeatingFrames))

        for i in range(0, Lt):
            pind = self.FindPeaks(S[i, :], self.SimilarityThreshold,
                                  self.MinDistanceBetweenFrames, self.MaxRepeatingFrames)
            I[i, :] = pind

        return I

    def FindPeaks(self, data, min_thr=0.5, min_dist=None, max_num=1):
        """
        Receives a row vector array of positive numerical values (in [0,1]) and finds the peak values and corresponding
         indices.
        
        Parameters:
            data (np.array): row vector of real values (in [0,1])
            min_thr: (Optional[float]) minimum threshold (in [0,1]) on data values. Defaults to 0.5
            min_dist:(Optional[int]) minimum distance (in # of elements) between peaks. Defaults to .25 * data.length
            max_num: (Optional[int]) maximum number of peaks. Defaults to 1
        
        Returns:
            Pi (np.array): sorted array of indices of peaks in the data
        """

        # make sure data is a Numpy matrix
        data = np.mat(data)

        lenData = data.shape[1]
        if min_dist is None:
            min_dist = np.floor(lenData / 4)

        peak_indices = np.zeros((1, max_num), int)

        data = np.multiply(data, (data >= min_thr))
        if np.size(np.nonzero(data)) < max_num:
            raise ValueError('not enough number of peaks! change parameters.')
        else:
            i = 0
            while i < max_num:
                peak_indices[0, i] = np.argmax(data)
                data[0, peak_indices[0, i] - min_dist - 1:peak_indices[0, i] + min_dist + 1] = 0
                i += 1
                if np.sum(data) == 0:
                    break

        peak_indices = np.sort(peak_indices)

        return peak_indices

    @staticmethod
    def ComputeRepeatingMaskSim(V, I):
        """Computes the soft mask for the repeating part using the magnitude spectrogram and the similarity indices

        Parameters:
            V (np.array): 2D matrix containing the magnitude spectrogram of a signal (Lf by Lt)
            I (np.array): array containing similarity indices for all time frames
        Returns:
            M (np.array): 2D matrix (Lf by Lt) containing the soft mask for the repeating part. Elements of M take on
            values in [0,1]
         """

        Lf, Lt = np.shape(V)
        W = np.zeros_like(V).T
        for i in range(0, Lt):
            pind = I[i, :]
            W[i, :] = np.median(V.T[pind.astype(int), :], axis=0)

        W = W.T
        Wrow = np.reshape(W, (1, Lf * Lt))
        Vrow = np.reshape(V, (1, Lf * Lt))
        W = np.min(np.vstack([Wrow, Vrow]), axis=0)
        W = np.reshape(W, (Lf, Lt))
        M = (W + Constants.EPSILON) / (V + Constants.EPSILON)

        return M

    @staticmethod
    def ComputeBeatSpectrum(X):
        """Computes the beat spectrum; averages (over freq.s) the autocorrelation matrix of a one-sided spectrogram.

         The autocorrelation matrix is computed by taking the autocorrelation of each row of the spectrogram and
         dismissing the symmetric half.

        Parameters:
            X (np.array): 2D matrix containing the one-sided power spectrogram of the audio signal (Lf by Lt)
        Returns:
            b (np.array): array containing the beat spectrum based on the power spectrogram
        """
        # compute the row-wise autocorrelation of the input spectrogram
        Lf, Lt = X.shape
        X = np.hstack([X, np.zeros_like(X)])
        Sx = np.abs(scifft.fft(X, axis=1) ** 2)  # fft over columns (take the fft of each row at a time)
        Rx = np.real(scifft.ifft(Sx, axis=1)[:, 0:Lt])  # ifft over columns
        NormFactor = np.tile(np.arange(1, Lt + 1)[::-1], (Lf, 1))  # normalization factor
        Rx = Rx / NormFactor

        # compute the beat spectrum
        b = np.mean(Rx, axis=0)  # average over frequencies

        return b

    @staticmethod
    def FindRepeatingPeriod(beat_spectrum, min_period, max_period):
        """Computes the repeating period of the sound signal using the beat spectrum.

        Parameters:
            beat_spectrum (np.array): input beat spectrum array
            min_period (int): minimum possible period value
            max_period (int): maximum possible period value
        Returns:
             period (int) : The period of the sound signal
        """

        beat_spectrum = beat_spectrum[1:]  # discard the first element of beat_spectrum (lag 0)
        beat_spectrum = beat_spectrum[min_period - 1:  max_period]
        period = np.argmax(beat_spectrum) + min_period  # TODO: not sure about this part

        return period

    @staticmethod
    def ComputeRepeatingMaskBeat(V, p):
        """Computes the soft mask for the repeating part using the magnitude spectrogram and the repeating period

        Parameters:
            V (np.array): 2D matrix containing the magnitude spectrogram of a signal (Lf by Lt)
            p (int): repeating period measured in # of time frames
        Returns:
            M (np.array): 2D matrix (Lf by Lt) containing the soft mask for the repeating part, elements of M take on
            values in [0,1]

        """

        Lf, Lt = V.shape
        r = np.ceil(float(Lt) / p)
        W = np.hstack([V, float('nan') * np.zeros((Lf, r * p - Lt))])
        W = np.reshape(W.T, (r, Lf * p))
        W1 = np.median(W[0:r, 0:Lf * (Lt - (r - 1) * p)], axis=0)
        W2 = np.median(W[0:r - 1, Lf * (Lt - (r - 1) * p):Lf * p], axis=0)
        W = np.hstack([W1, W2])
        W = np.reshape(np.tile(W, (r, 1)), (r * p, Lf)).T
        W = W[:, 0:Lt]

        Wrow = W.flatten() # np.reshape(W, (1, Lf * Lt))
        Vrow = V.flatten() # np.reshape(V, (1, Lf * Lt))
        W = np.min(np.vstack([Wrow, Vrow]), axis=0)
        W = np.reshape(W, (Lf, Lt))
        M = (W + Constants.EPSILON) / (V + Constants.EPSILON)

        return M

    def _updatePeriod(self, period):
        period = float(period)
        result = period * self.Mixture.SampleRate
        result += self.WindowAttributes.WindowLength / self.WindowAttributes.WindowOverlap - 1
        result /= self.WindowAttributes.WindowOverlap
        return np.ceil(result)

    def Plot(self, outputFile):
        """ NOT YET IMPLEMENTED. Plots REPET results and saves to file.

        Raises:
            NotImplementedError

        Args:

        Returns:

        """
        raise NotImplementedError('You shouldn\'t be calling this yet...')

    def MakeAudioSignals(self):
        """ Returns the background and foreground audio signals

        Returns:
            Audio Signals (List): 2 element list.

                * Background: Audio signal with the calculated background track
                * Foreground: Audio signal with the calculated foreground track

        """
        self.fgnd = self.Mixture - self.bkgd
        return [self.bkgd, self.fgnd]


class RepetType():
    """Types of Repet algorithm implementation.
    """
    ORIGINAL = 'original'
    SIM = 'sim'
    ADAPTIVE = 'adaptive'
    DEFAULT = ORIGINAL

    def __init__(self):
        pass
