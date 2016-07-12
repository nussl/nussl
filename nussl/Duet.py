#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy import signal

import spectral_utils
import separation_base
import audio_signal
import constants


class Duet(separation_base.SeparationBase):
    """Implements the Degenerate Unmixing Estimation Technique (DUET) algorithm.

    The DUET algorithm was originally proposed by S.Rickard and F.Dietrich for DOA estimation
    and further developed for BSS and demixing by A.Jourjine, S.Rickard, and O. Yilmaz.


    References:

        * Rickard, Scott. "The DUET blind source separation algorithm." Blind Speech Separation. Springer Netherlands,
          2007. 217-241.
        * Yilmaz, Ozgur, and Scott Rickard. "Blind separation of speech mixtures via time-frequency masking." Signal
          Processing, IEEE transactions on 52.7 (2004): 1830-1847.

    Parameters:
        audio_signal (np.array): a 2-row Numpy matrix containing samples of the two-channel mixture
        num_sources (int): number of sources to find
        a_min (Optional[int]): Minimum attenuation. Defaults to -3
        a_max(Optional[int]): Maximum attenuation. Defaults to 3
        a_num(Optional[int]): Number of bins for attenuation. Defaults to 50
        d_min (Optional[int]): Minimum delay. Defaults to -3
        d_max (Optional[int]): Maximum delay. Defaults to 3
        d_num (Optional[int]): Number of bins for delay. Defaults to 50
        threshold (Optional[float]): Value in [0, 1] for peak picking. Defaults to 0.2
        a_min_distance (Optional[int]): Minimum distance between peaks wrt attenuation. Defaults to 5
        d_min_distance (Optional[int]): Minimum distance between peaks wrt delay. Defaults to 5
        stft_params (Optional[WindowAttributes]): Window attributes for stft. Defaults to
         WindowAttributes.WindowAttributes(self.sample_rate)
        sample_rate (Optional[int]): Sample rate for the audio. Defaults to Constants.DEFAULT_SAMPLE_RATE

    Examples:
        :ref:`The DUET Demo Example <duet_demo>`

    """

    def __init__(self, input_audio_signal, num_sources,
                 a_min=-3, a_max=3, a_num=50, d_min=-3, d_max=3, d_num=50,
                 threshold=0.2, a_min_distance=5, d_min_distance=5):
        # TODO: Is there a better way to do this?
        self.__dict__.update(locals())
        super(Duet, self).__init__(input_audio_signal)
        self.separated_sources = None
        self.a_grid = None
        self.d_grid = None
        self.hist = None

    def __str__(self):
        return 'Duet'

    def run(self):
        """Extracts N sources from a given stereo audio mixture (N sources captured via 2 sensors)

        Returns:
            * **xhat** (*np.array*) - an N-row matrix containing time-domain estimates of sources
            * **ad_est** (*np.array*) - N by 2 Numpy matrix containing estimated attenuation and delay values
              corresponding to N sources
        Example:
             ::
            input_file_name = '../Input/dev1_female3_inst_mix.wav'
            signal = AudioSignal(path_to_input_file=input_file_name)

            duet = Duet(signal, a_min=-3, a_max=3, a_num=50, d_min=-3, d_max=3, d_num=50, threshold=0.2,
            a_min_distance=5, d_min_distance=5, num_sources=3)

            duet.run()

        """

        if self.audio_signal.num_channels != 2:
            raise Exception('Cannot run DUET on audio signal without exactly 2 channels!')

        # Give them shorter names
        L = self.stft_params.window_length
        winType = self.stft_params.window_type
        hop = self.stft_params.hop_length
        fs = self.sample_rate

        # Compute the stft of the two channel mixtures
        stft1, psd1, frequency_vector, time_vector = spectral_utils.e_stft_plus(self.audio_signal.get_channel(1), L, hop, winType, fs, use_librosa=True)
        stft2, psd2, frequency_vector, time_vector = spectral_utils.e_stft_plus(self.audio_signal.get_channel(2), L, hop, winType, fs, use_librosa=True)

        # remove dc component to avoid dividing by zero freq. in the delay estimation
        stft1 = stft1[1::, :]
        stft2 = stft2[1::, :]
        num_frequency_bins = len(frequency_vector)
        num_time_bins = len(time_vector)

        # Compute the freq. matrix for later use in phase calculations
        wmat = np.array(np.tile(np.mat(frequency_vector[1::]).T, (1, num_time_bins))) * (2 * np.pi / fs)  # WTF?

        # Calculate the symmetric attenuation (alpha) and delay (delta) for each
        # time-freq. point
        R21 = (stft2 + constants.EPSILON) / (stft1 + constants.EPSILON)
        atn = np.abs(R21)  # relative attenuation between the two channels
        alpha = atn - 1 / atn  # symmetric attenuation
        delta = -np.imag(np.log(R21)) / (2 * np.pi * wmat)  # relative delay

        # What is going on here???
        # ------------------------------------
        # calculate the weighted histogram
        p = 1
        q = 0
        tfw = (np.abs(stft1) * np.abs(stft2)) ** p * (np.abs(wmat)) ** q  # time-freq weights

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
        self.non_normalized_hist = H[0]
        self.smoothed_hist = self.twoDsmooth(self.non_normalized_hist, np.array([3]))

        # smooth the histogram - local average 3-by-3 neighboring bins
        hist = self.twoDsmooth(hist, np.array([3]))

        # normalize and plot the histogram
        hist /= hist.max()

        # find the location of peaks in the alpha-delta plane
        self.peak_indices = self.find_peaks2(hist, self.threshold,
                                  np.array([self.a_min_distance, self.d_min_distance]), self.num_sources)

        alphapeak = agrid[self.peak_indices[0, :]]
        deltapeak = dgrid[self.peak_indices[1, :]]

        ad_est = np.vstack([alphapeak, deltapeak]).T

        # convert alpha to a
        atnpeak = (alphapeak + np.sqrt(alphapeak ** 2 + 4)) / 2

        # compute masks for separation
        bestsofar = np.inf * np.ones((num_frequency_bins - 1, num_time_bins))
        bestind = np.zeros((num_frequency_bins - 1, num_time_bins), int)
        for i in range(0, self.num_sources):
            score = np.abs(atnpeak[i] * np.exp(-1j * wmat * deltapeak[i]) * stft1 - stft2) ** 2 / (1 + atnpeak[i] ** 2)
            mask = (score < bestsofar)
            bestind[mask] = i
            bestsofar[mask] = score[mask]

        # demix with ML alignment and convert to time domain
        Lx = self.audio_signal.signal_length
        xhat = np.zeros((self.num_sources, Lx))
        for i in range(0, self.num_sources):
            mask = (bestind == i)
            Xm = np.vstack([np.zeros((1, num_time_bins)),
                            (stft1 + atnpeak[i] * np.exp(1j * wmat * deltapeak[i]) * stft2) / (1 + atnpeak[i] ** 2) * mask])
            # xi = spectral_utils.f_istft(Xm, L, winType, hop, fs)
            xi = spectral_utils.e_istft(Xm, L, hop, winType)

            xhat[i, :] = xi[0:Lx]
            # add back to the separated signal a portion of the mixture to eliminate
            # most of the masking artifacts
            # xhat=xhat+0.05*x[0,:]

        self.separated_sources = xhat

        return xhat, ad_est

    @staticmethod
    def find_peaks2(data, min_thr=0.5, min_dist=None, max_peaks=1):
        """Receives a matrix of positive numerical values (in [0,1]) and finds the peak values and corresponding indices.

        Parameters:
            data (np.array): a 2D Numpy matrix containing real values (in [0,1])
            min_thr (Optional[float]): minimum threshold (in [0,1]) on data values. Defaults to 0.5
            min_dist (Optional[np.array]): 1 by 2 matrix containing minimum distances (in # of time elements) between
             peaks row-wise and column-wise. Defaults to .25* matrix dimensions
            max_peaks (Optional[int]): maximum number of peaks in the whole matrix. Defaults to 1

        Returns:
            peakIndex (np.array): a two-row Numpy matrix containing peaks and their indices
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
        """Receives a row vector of positive values (in [0,1]) and finds the peak values and corresponding indices.

       Parameters:
            data (np.array): a 2D Numpy matrix containing real values (in [0,1])
            min_thr (Optional[float]): minimum threshold (in [0,1]) on data values. Defaults to 0.5
            min_dist (Optional[np.array]): 1 by 2 matrix containing minimum distances (in # of time elements) between
             peaks row-wise and column-wise. Defaults to .25* matrix dimensions
            max_peaks (Optional[int]): maximum number of peaks in the whole matrix. Defaults to 1

        Returns:
            peakIndex (np.array): a two-row Numpy matrix containing peaks and their indices
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
        """Performs two-dimensional convolution in order to smooth the values of matrix elements.

        (similar to low-pass filtering)

        Parameters:
            Mat (np.array): a 2D Numpy matrix to be smoothed
            Kernel (np.array): a 2D Numpy matrix containing kernel values
        Note:
            if Kernel is of size 1 by 1 (scalar), a Kernel by Kernel matrix of 1/Kernel**2 will be used as the matrix
            averaging kernel
        Output:
            SMat (np.array): a 2D Numpy matrix containing a smoothed version of Mat (same size as Mat)
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
        """Returns the extracted signals

        Returns:
            signals (List[AudioSignal]): List of AudioSignals extracted using DUET.

        """
        signals = []
        for i in range(self.num_sources):
            cur_signal = audio_signal.AudioSignal(audio_data_array=self.separated_sources[i],
                                                  sample_rate=self.sample_rate)
            signals.append(cur_signal)
        return signals

    def plot(self, output_name, three_d_plot=False, normalize=True):
        """Plots histograms with the results of the DUET algorithm

        Parameters:
            output_name (str): path to save plot as
            three_d_plot (Optional[bool]): Flags whether or not to plot in 3d. Defaults to False
        """
        plt.close('all')

        AA = np.tile(self.a_grid[1::], (self.d_num, 1)).T
        DD = np.tile(self.d_grid[1::].T, (self.a_num, 1))

        histogram_data = self.hist if normalize else self.smoothed_hist

        # plot the histogram in 2D
        if not three_d_plot:
            plt.figure()
            plt.pcolormesh(AA, DD, histogram_data)
            plt.xlabel(r'$\alpha$', fontsize=16)
            plt.ylabel(r'$\delta$', fontsize=16)
            plt.title(r'$\alpha-\delta$ Histogram')
            plt.axis('tight')
            plt.savefig(output_name)
            plt.close()

        else:
            # plot the histogram in 3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_wireframe(AA, DD, histogram_data, rstride=2, cstride=2)
            plt.xlabel(r'$\alpha$', fontsize=16)
            plt.ylabel(r'$\delta$', fontsize=16)
            plt.title(r'$\alpha-\delta$ Histogram')
            plt.axis('tight')
            ax.view_init(30, 30)
            plt.savefig(output_name)
            plt.close()
