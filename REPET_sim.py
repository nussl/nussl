"""
This module implements the REpeating Pattern Extraction Technique algorithm using the 
Similarity Matrix (REPET-SIM). REPET is a simple method for separating the repeating 
background from the non-repeating foreground in a piece of audio mixture. 
REPET-SIM is a generalization of REPET, which looks for similarities instead of 
periodicities.

See http://music.eecs.northwestern.edu/research.php?project=repet

References:
[1] Zafar Rafii and Bryan Pardo. "Audio Separation System and Method," 
    US20130064379 A1, US 13/612,413, March 14, 2013.
[2] Zafar Rafii and Bryan Pardo. 
    "Music/Voice Separation using the Similarity Matrix," 
    13th International Society on Music Information Retrieval, 
    Porto, Portugal, October 8-12, 2012.


Required packages:
1. Numpy
2. Scipy
3. Matplotlib

Required modules:
1. f_stft
2. f_istft
3. time
"""

import numpy as np
from FftUtils import f_stft
from f_istft import f_istft
# import matplotlib.pyplot as plt
import SeparationBase, Constants


# plt.interactive('True')


class RepetSim(SeparationBase.SeparationBase):
    def __init__(self, windowAttributes=None, sampleRate=None, audioSignal=None,
                 similarityThreshold=0, similarityDistance=1, similarityMaxRepetitions=100):
        super(RepetSim, self).__init__(windowAttributes=windowAttributes, sampleRate=sampleRate,
                                       audioSignal=audioSignal)
        self.SimilarityThreshold = similarityThreshold
        self.SimilarityDistance = similarityDistance
        self.SimilarityMaxRepetitions = similarityMaxRepetitions

    # @property
    def Run(self):
        """
        The repet_sim function implements the main algorithm.
        
        Inputs:
        x: audio mixture (M by N) containing M channels and N time samples
        fs: sampling frequency of the audio signal
        specparam (optional): list containing STFT parameters including in order the window length, 
                              window type, overlap in # of samples, and # of fft points.
                              default: window length: 40 mv
                                       window type: Hamming
                                       overlap: window length/2
                                       nfft: window length
        par: (optional) Numpy array containing similarity parameters (3 values) (default: [0,1,100])
              -- par[0]: minimum threshold (in [0,1]) for the similarity measure 
                  within repeating frames
              -- par[1]: minimum distance (in seconds) between repeating frames
              -- par[2]: maximum number of repeating frames for the median filter 
             
        Output:
        y: repeating background (M by N) containing M channels and N time samples
           (the corresponding non-repeating foreground is equal to x-y)
           
        EXAMPLE:
        x,fs,enc = wavread('mixture.wav'); 
        y = repet_sim(np.mat(x),fs,np.array([0,1,100]))
        wavwrite(y,'background.wav',fs,enc)
        wavwrite(x-y,'foreground.wav',fs,enc)

        * Note: the 'scikits.audiolab' package is required for reading and writing 
                .wav files
        """

        par = np.array([self.SimilarityThreshold, self.SimilarityDistance, self.SimilarityMaxRepetitions])

        # STFT parameters
        L, win, ovp, nfft = self.WindowAttributes.WindowLength, self.WindowAttributes.WindowType, \
                            self.WindowAttributes.WindowOverlap, self.WindowAttributes.Nfft



        # HPF parameters
        fc = 100  # cutoff freqneyc (in Hz) of the high pass filter
        fc = np.ceil(
            float(fc) * (self.WindowAttributes.Nfft - 1) / self.SampleRate)  # cutoff freq. (in # of freq. bins)

        # compute the spectrograms of all channels
        M, N = np.shape(self.Mixture.AudioData)
        X = f_stft(np.mat(self.Mixture.AudioData[0, :]), self.WindowAttributes.WindowLength,
                   self.WindowAttributes.WindowType, ovp, self.SampleRate, nfft, 0)[0]

        print "f_stft done"
        for i in range(1, M):
            Sx = f_stft(np.mat(self.Mixture.AudioData[i, :]), L, win, ovp, self.SampleRate, nfft, 0)[0]
            X = np.dstack([X, Sx])
        V = np.abs(X)
        if M == 1:
            X = X[:, :, np.newaxis]
            V = V[:, :, np.newaxis]

        Vavg = np.mean(V, axis=2)
        S = self.ComputeSimilarityMatrix(Vavg)
        print "ComputeSimilarityMatrix done"

        # plot the similarity matrix 
        # plt.figure()
        # plt.pcolormesh(S)
        # plt.axis('tight')
        # plt.title('Similarity Matrix')

        self.SimilarityDistance = np.round(self.SimilarityDistance * self.SampleRate / ovp)
        S = self.FindSimilarityIndices(S)
        print "FindSimilarityIndices done"

        # separate the mixture background by masking
        y = np.zeros((M, N))
        for i in range(0, M):
            RepMask = self.ComputeRepeatingMask(V[:, :, i], S)
            RepMask[1:fc, :] = 1  # high-pass filter the foreground
            XMi = RepMask * X[:, :, i]
            yi = f_istft(XMi, L, win, ovp, self.SampleRate)[0]
            y[i, :] = yi[0:N]

        return y

    def ComputeSimilarityMatrix(self, X):
        """
        Computes the similarity matrix using the cosine similarity.
        
        Input:
        X: 2D matrix containing the magnitude spectrogram of the audio signal (Lf by Lt)
        Output:
        S: similarity matrix (Lt by Lt)
        """
        # normalize the columns of the magnitude spectrogram
        Lt = np.shape(X)[1]
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
        
        Inputs:
        S: similarity matrix (Lt by Lt)
        simparam: array containing 3 similarity parameters 
              -- simparam[0]: minimum threshold (in [0,1]) for the similarity measure 
                  within repeating frames
              -- simparam[1]: minimum distance (in # of time frames) between repeating frames
              -- simparam[2]: maximum number of repeating frames for the median filter   
                 
        Output:
        I: array containing similarity indices for all time frames
        """

        Lt = np.shape(S)[0]
        I = np.zeros((Lt, self.SimilarityMaxRepetitions))

        for i in range(0, Lt):
            pind = self.FindPeaks(S[i, :], self.SimilarityThreshold,
                                  self.SimilarityDistance, self.SimilarityMaxRepetitions)
            I[i, :] = pind

        return I

    def FindPeaks(self, data, min_thr=0.5, min_dist=None, max_num=1):
        """
        The 'FindPeaks' function receives a row vector array of positive numerical
        values (in [0,1]) and finds the peak values and corresponding indices.
        
        Inputs: 
        data: row vector of real values (in [0,1])
        min_thr: (optional) minimum threshold (in [0,1]) on data values - default=0.5
        min_dist:(optional) minimum distance (in # of time elements) between peaks
                 default: 25% of the vector length
        max_num: (optional) maximum number of peaks - default: 1
        
        Output:
        Pi: peaks indices
        """

        # make sure data is a Numpy matrix
        data = np.mat(data)

        lenData = np.shape(data)[1]
        if min_dist is None:
            min_dist = np.floor(lenData / 4)

        Pi = np.zeros((1, max_num), int)

        data = np.multiply(data, (data >= min_thr))
        if np.size(np.nonzero(data)) < max_num:
            raise ValueError('not enough number of peaks! change parameters.')
        else:
            i = 0
            while i < max_num:
                Pi[0, i] = np.argmax(data)
                data[0, Pi[0, i] - min_dist - 1:Pi[0, i] + min_dist + 1] = 0
                i += 1
                if np.sum(data) == 0:
                    break

        Pi = np.sort(Pi)

        return Pi

    def ComputeRepeatingMask(self, V, I):
        """
            Computes the soft mask for the repeating part using
            the magnitude spectrogram and the similarity indices

            Inputs:
            V: 2D matrix containing the magnitude spectrogram of a signal (Lf by Lt)
            I: array containing similarity indices for all time frames
            Output:
            M: 2D matrix (Lf by Lt) containing the soft mask for the repeating part,
            elements of M take on values in [0,1]
         """

        Lf, Lt = np.shape(V)
        W = np.zeros((Lt, Lf))
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
