#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy import signal

import FftUtils
import SeparationBase
import AudioSignal
import Constants


class Duet(SeparationBase.SeparationBase):
    """
    This module implements the Degenerate Unmixing Estimation Technique (DUET) algorithm.
    The DUET algorithm was originally proposed by S.Rickard and F.Dietrich for DOA estimation
    and further developed for BSS and demixing by A.Jourjine, S.Rickard, and O. Yilmaz.

    References:
    [1] Rickard, Scott. "The DUET blind source separation algorithm." Blind Speech Separation.
        Springer Netherlands, 2007. 217-241.
    [2] Yilmaz, Ozgur, and Scott Rickard. "Blind separation of speech mixtures via
        time-frequency masking." Signal Processing, IEEE transactions on 52.7
        (2004): 1830-1847.

    Authors: Fatameh Pishdadian and Ethan Manilow
    Interactive Audio Lab
    Northwestern University, 2015

    """
    def __init__(self, audio_signal, num_sources, a_min=-3, a_max=3, a_num=50, d_min=-3, d_max=3, d_num=50,
                  threshold=0.2, a_min_distance=5, d_min_distance=5, window_attributes=None, sample_rate=None):
        # TODO: Is there a better way to do this?
        self.__dict__.update(locals())
        super(Duet, self).__init__(window_attributes, sample_rate, audio_signal)
        self.separated_sources = None
        self.a_grid = None
        self.d_grid = None
        self.hist = None

    def __str__(self):
        return 'Duet'

    def run(self):
        """
        The 'duet' function extracts N sources from a given stereo audio mixture
        (N sources captured via 2 sensors)

        Inputs:
        x: a 2-row Numpy matrix containing samples of the two-channel mixture
        sparam: structure array containing spectrogram parameters including
                L: window length (in # of samples)
              win: window type, string ('Rectangular', 'Hamming', 'Hanning', 'Blackman')
              ovp: number of overlapping samples between adjacent windows
              num_fft_bins: min number of desired freq. samples in (-pi,pi]. MUST be >= L.
                   *NOTE* If this is not a power of 2, then it will automatically
                   zero-pad up to the next power of 2. IE if you put 257 here,
                   it will pad up to 512.
               fs: sampling rate of the signal
               ** sparam = np.array([(L,win,ovp,num_fft_bins,fs)]
               dtype=[('winlen',int),('wintype','|S10'),('overlap',int),('numfreq',int),('sampfreq',int)])

        adparam: structure array containing ranges and number of bins for attenuation and delay
               ** adparam = np.array([(self.aMin,self.aMax,self.a_num,self.dMin,self.dMax,self.d_num)],
               dtype=[('amin',float),('amax',float),('anum',float),('dmin',float)
               ,('dmax',float),('dnum',int)])

        Pr: vector containing user defined information including a threshold value (in [0,1])
            for peak picking (thr), minimum distance between peaks, and the number of sources (N)
            ** Pr = np.array(thr,self.aMindist,self.dMindist,N)
        plothist: (optional) string input, indicates if the histogram is to be plotted
              'y' (default): plot the histogram, 'n': don't plot

        Output:
        xhat: an N-row Numpy matrix containing N time-domain estimates of sources
        ad_est: N by 2 Numpy matrix containing estimated attenuation and delay values
              corresponding to N sources
        """

        if self.audio_signal.num_channels != 2:
            raise Exception('Cannot run DUET on audio signal without exactly 2 channels!')


        # Give them shorter names
        L = self.window_attributes.window_length
        winType = self.window_attributes.window_type
        ovp = self.window_attributes.window_overlap
        fs = self.sample_rate

        # Compute the do_STFT of the two channel mixtures
        X1, P1, F, T = FftUtils.f_stft(self.audio_signal.get_channel(1), window_attributes=self.window_attributes,
                                       sample_rate=fs)
        X2, P2, F, T = FftUtils.f_stft(self.audio_signal.get_channel(2), window_attributes=self.window_attributes,
                                       sample_rate=fs)

        # remove dc component to avoid dividing by zero freq. in the delay estimation
        X1 = X1[1::, :]
        X2 = X2[1::, :]
        Lf = len(F)
        Lt = len(T)

        # Compute the freq. matrix for later use in phase calculations
        wmat = np.array(np.tile(np.mat(F[1::]).T, (1, Lt))) * (2 * np.pi / fs)  # WTF?

        # Calculate the symmetric attenuation (alpha) and delay (delta) for each
        # time-freq. point
        R21 = (X2 + Constants.EPSILON) / (X1 + Constants.EPSILON)
        atn = np.abs(R21)  # relative attenuation between the two channels
        alpha = atn - 1 / atn  # symmetric attenuation
        delta = -np.imag(np.log(R21)) / (2 * np.pi * wmat)  # relative delay

        # What is going on here???
        # ------------------------------------
        # calculate the weighted histogram
        p = 1
        q = 0
        tfw = (np.abs(X1) * np.abs(X2)) ** p * (np.abs(wmat)) ** q  # time-freq weights

        # only consider time-freq. points yielding estimates in bounds
        a_premask = np.logical_and(self.a_min < alpha, alpha < self.a_max)
        d_premask = np.logical_and(self.d_min < delta, delta < self.d_max)
        ad_premask = np.logical_and(a_premask, d_premask)

        ad_nzind = np.nonzero(ad_premask)
        alpha_vec = alpha[ad_nzind]
        delta_vec = delta[ad_nzind]
        tfw_vec = tfw[ad_nzind]

        # compute the histogram
        H = np.histogram2d(alpha_vec, delta_vec, bins=np.array([self.a_num, self.d_num]),
                           range=np.array([[self.a_min, self.a_max], [self.d_min, self.d_max]]), normed=False,
                           weights=tfw_vec)

        hist = H[0] / H[0].max()
        agrid = H[1]
        dgrid = H[2]

        # Save these for later
        self.a_grid = agrid
        self.d_grid = dgrid
        self.hist = hist

        # smooth the histogram - local average 3-by-3 neighboring bins
        hist = self.twoDsmooth(hist, np.array([3]))

        # normalize and plot the histogram
        hist /= hist.max()

        # find the location of peaks in the alpha-delta plane
        pindex = self.find_peaks2(hist, self.threshold,
                                  np.array([self.a_min_distance, self.d_min_distance]), self.num_sources)

        alphapeak = agrid[pindex[0, :]]
        deltapeak = dgrid[pindex[1, :]]

        ad_est = np.vstack([alphapeak, deltapeak]).T

        # convert alpha to a
        atnpeak = (alphapeak + np.sqrt(alphapeak ** 2 + 4)) / 2

        # compute masks for separation
        bestsofar = np.inf * np.ones((Lf - 1, Lt))
        bestind = np.zeros((Lf - 1, Lt), int)
        for i in range(0, self.num_sources):
            score = np.abs(atnpeak[i] * np.exp(-1j * wmat * deltapeak[i]) * X1 - X2) ** 2 / (1 + atnpeak[i] ** 2)
            mask = (score < bestsofar)
            bestind[mask] = i
            bestsofar[mask] = score[mask]

        # demix with ML alignment and convert to time domain
        Lx = self.audio_signal.signal_length
        xhat = np.zeros((self.num_sources, Lx))
        for i in range(0, self.num_sources):
            mask = (bestind == i)
            Xm = np.vstack([np.zeros((1, Lt)), (X1 + atnpeak[i] * np.exp(1j * wmat * deltapeak[i]) * X2)
                            / (1 + atnpeak[i] ** 2) * mask])
            xi = FftUtils.f_istft(Xm, L, winType, ovp, fs)

            xhat[i, :] = np.array(xi)[0, 0:Lx]
            # add back to the separated signal a portion of the mixture to eliminate
            # most of the masking artifacts
            # xhat=xhat+0.05*x[0,:]

        self.separated_sources = xhat

        return xhat, ad_est

    @staticmethod
    def find_peaks2(data, min_thr=0.5, min_dist=None, max_peaks=1):

        """
        The 'find_peaks2d' function receives a matrix of positive numerical
        values (in [0,1]) and finds the peak values and corresponding indices.

        Inputs:
        data: a 2D Numpy matrix containing real values (in [0,1])
        min_thr:(optional) minimum threshold (in [0,1]) on data values - default=0.5
        min_dist:(optional) 1 by 2 matrix containing minimum distances (in # of time elements) between peaks
                  row-wise and column-wise - default: 25% of matrix dimensions
        max_peaks: (optional) maximum number of peaks in the whole matrix - default: 1

        Output:
        Pi: a two-row Numpy matrix containing peaks indices
        """
        assert (type(data) == np.ndarray)

        Rdata, Cdata = data.shape
        if min_dist is None:
            min_dist = np.array([np.floor(Rdata / 4), np.floor(Cdata / 4)])

        peakIndex = np.zeros((2, max_peaks), int)
        Rmd, Cmd = min_dist.astype(int)

        # keep only the values that pass the threshold
        data = np.multiply(data, (data >= min_thr))

        if np.size(np.nonzero(data)) < max_peaks:
            raise ValueError('not enough number of peaks! change parameters.')
        else:
            i = 0
            while i < max_peaks:
                peakIndex[:, i] = np.unravel_index(data.argmax(), data.shape)
                n1 = peakIndex[0, i] - Rmd - 1
                m1 = peakIndex[0, i] + Rmd + 1
                n2 = peakIndex[1, i] - Cmd - 1
                m2 = peakIndex[1, i] + Cmd + 1
                data[n1:m1, n2:m2] = 0
                i += 1
                if np.sum(data) == 0:
                    break

        return peakIndex

    @staticmethod
    def find_peaks(data, min_thr=0.5, min_dist=None, max_num=1):
        # TODO: consolidate both find_peaks functions
        # TODO: when is this used?
        """
        The 'find_peaks' function receives a row vector array of positive numerical
        values (in [0,1]) and finds the peak values and corresponding indices.

        Inputs:
        data: row vector of real values (in [0,1])
        min_thr: (optional) minimum threshold (in [0,1]) on data values - default=0.5
        min_dist:(optiotnal) minimum distance (in # of time elements) between peaks
                 default: 25% of the vector length
        max_num: (optional) maximum number of peaks - default: 1

        Output:
        Pi: peaks indices
        """
        assert (type(data) == np.ndarray)

        if min_dist is None:
            min_dist = np.floor(data.shape[1] / 4)

        peak_indices = np.zeros((1, max_num), int)

        # keep only values above the threshold
        data = np.multiply(data, (data >= min_thr))

        if np.size(np.nonzero(data)) < max_num:
            raise ValueError('not enough number of peaks! change parameters.')
        else:
            i = 0
            while i < max_num:
                peak_indices[0, i] = np.argmax(data)
                n = peak_indices[0, i] - min_dist - 1
                m = peak_indices[0, i] + min_dist + 1
                data[0, n: m] = 0
                i += 1
                if np.sum(data) == 0:
                    break

        peak_indices = np.sort(peak_indices)

        return peak_indices

    def twoDsmooth(self, Mat, Kernel):
        """
        The 'twoDsmooth' function receives a matrix and a kernel type and performs
        two-dimensional convolution in order to smooth the values of matrix elements.
        (similar to low-pass filtering)

        Inputs:
        Mat: a 2D Numpy matrix to be smoothed
        Kernel: a 2D Numpy matrix containing kernel values
               Note: if Kernel is of size 1 by 1 (scalar), a Kernel by Kernel matrix
               of 1/Kernel**2 will be used as the matrix averaging kernel
        Output:
        SMat: a 2D Numpy matrix containing a smoothed version of Mat (same size as Mat)
        """

        # check the dimensions of the Kernel matrix and set the values of the averaging
        # matrix, Kmat
        if np.prod(Kernel.shape) == 1:
            Kmat = np.ones((Kernel, Kernel)) / Kernel ** 2
        else:
            Kmat = Kernel

        # make Kmat have odd dimensions
        krow, kcol = np.shape(Kmat)
        if np.mod(krow, 2) == 0:
            Kmat = signal.convolve2d(Kmat, np.ones((2, 1))) / 2
            krow += 1

        if np.mod(kcol, 2) == 0:
            Kmat = signal.convolve2d(Kmat, np.ones((1, 2))) / 2
            kcol += 1

        # adjust the matrix dimension for convolution
        copyrow = int(np.floor(krow / 2))  # number of rows to copy on top and bottom
        copycol = int(np.floor(kcol / 2))  # number of columns to copy on either side

        # TODO: This is very ugly. Make this readable
        # form the augmented matrix (rows and columns added to top, bottom, and sides)
        Mat = np.mat(Mat)  # make sure Mat is a Numpy matrix
        augMat = np.vstack(
            [
                np.hstack(
                    [Mat[0, 0] * np.ones((copyrow, copycol)),
                     np.ones((copyrow, 1)) * Mat[0, :],
                     Mat[0, -1] * np.ones((copyrow, copycol))
                     ]),
                np.hstack(
                    [Mat[:, 0] * np.ones((1, copycol)),
                     Mat,
                     Mat[:, -1] * np.ones((1, copycol))]),
                np.hstack(
                    [Mat[-1, 1] * np.ones((copyrow, copycol)),
                     np.ones((copyrow, 1)) * Mat[-1, :],
                     Mat[-1, -1] * np.ones((copyrow, copycol))
                     ]
                )
            ]
        )

        # perform two-dimensional convolution between the input matrix and the kernel
        SMAT = signal.convolve2d(augMat, Kmat[::-1, ::-1], mode='valid')

        return SMAT


    def make_audio_signals(self):
        signals = []
        for i in range(self.num_sources):
            signal = AudioSignal.AudioSignal(audio_data_array=self.separated_sources[i])
            signals.append(signal)
        return signals

    def plot(self, outputName, three_d_plot=False):
        plt.close('all')

        AA = np.tile(self.a_grid[1::], (self.d_num, 1)).T
        DD = np.tile(self.d_grid[1::].T, (self.a_num, 1))

        # plot the histogram in 2D
        if not three_d_plot:
            plt.figure()
            plt.pcolormesh(AA, DD, self.hist)
            plt.xlabel(r'$\alpha$', fontsize=16)
            plt.ylabel(r'$\delta$', fontsize=16)
            plt.title(r'$\alpha-\delta$ Histogram')
            plt.axis('tight')
            plt.savefig(outputName)
            plt.close()

        else:
            # plot the histogram in 3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_wireframe(AA, DD, self.hist, rstride=2, cstride=2)
            plt.xlabel(r'$\alpha$', fontsize=16)
            plt.ylabel(r'$\delta$', fontsize=16)
            plt.title(r'$\alpha-\delta$ Histogram')
            plt.axis('tight')
            ax.view_init(30, 30)
            plt.savefig(outputName)
            plt.close()
