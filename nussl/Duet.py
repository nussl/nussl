#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
# noinspection PyUnusedImport
from mpl_toolkits.mplot3d import axes3d
from scipy import signal

import spectral_utils
import separation_base
import audio_signal
import constants
import utils


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
        input_audio_signal (np.array): a 2-row Numpy matrix containing samples of the two-channel mixture
        num_sources (int): number of sources to find
        attenuation_min (Optional[int]): Minimum attenuation. Defaults to -3
        attenuation_max (Optional[int]): Maximum attenuation. Defaults to 3
        num_attenuation_bins (Optional[int]): Number of bins for attenuation. Defaults to 50
        delay_min (Optional[int]): Minimum delay. Defaults to -3
        delay_max (Optional[int]): Maximum delay. Defaults to 3
        num_delay_bins (Optional[int]): Number of bins for delay. Defaults to 50
        peak_threshold (Optional[float]): Value in [0, 1] for peak picking. Defaults to 0.2
        attenuation_min_distance (Optional[int]): Minimum distance between peaks wrt attenuation. Defaults to 5
        delay_min_distance (Optional[int]): Minimum distance between peaks wrt delay. Defaults to 5

    Examples:
        :ref:`The DUET Demo Example <duet_demo>`

    """

    def __init__(self, input_audio_signal, num_sources,
                 attenuation_min=-3, attenuation_max=3, num_attenuation_bins=50,
                 delay_min=-3, delay_max=3, num_delay_bins=50,
                 peak_threshold=0.2, attenuation_min_distance=5, delay_min_distance=5):
        super(Duet, self).__init__(input_audio_signal=input_audio_signal)

        if self.audio_signal.num_channels != 2:
            raise ValueError('Duet requires that the input_audio_signal has exactly 2 channels!')

        self.num_sources = num_sources
        self.attenuation_min = attenuation_min
        self.attenuation_max = attenuation_max
        self.num_attenuation_bins = num_attenuation_bins
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.num_delay_bins = num_delay_bins
        self.peak_threshold = peak_threshold
        self.attenuation_min_distance = attenuation_min_distance
        self.delay_min_distance = delay_min_distance
        self.separated_sources = None
        self.attenuation_grid = None
        self.delay_grid = None

        self.attenuation_delay_histogram = None
        self.non_normalized_hist = None
        self.smoothed_hist = None
        self.peak_indices = None

        self.symmetric_atn = None
        self.delay = None
        self.num_frequency_bins = None
        self.num_time_bins = None
        self.stft_ch0 = None
        self.stft_ch = None
        self.wmat = None

    def run(self):
        """Extracts N sources from a given stereo audio mixture (N sources captured via 2 sensors)

        Returns:
            * **source_estimates** (*np.array*) - an N-row matrix containing time-domain estimates of sources
            * **atn_delay_est** (*np.array*) - N by 2 Numpy matrix containing estimated attenuation and delay values
              corresponding to N sources
        Example:
             ::
            input_file_name = '../Input/dev1_female3_inst_mix.wav'
            signal = AudioSignal(path_to_input_file=input_file_name)

            duet = Duet(signal, a_min=-3, a_max=3, a_num=50, d_min=-3, d_max=3, d_num=50, threshold=0.2,
            a_min_distance=5, d_min_distance=5, num_sources=3)

            duet.run()

        """

        # Calculate the stft of both channels and create the frequency matrix
        self.stft_ch0, self.stft_ch1, self.wmat = self.compute_spectrogram(self.sample_rate)

        # Calculate the symmetric attenuation (alpha) and delay (delta) for each
        # time-freq. point
        self.symmetric_atn, self.delay = self.compute_atn_delay()

        # Make histogram and bins
        hist, agrid, dgrid = self.make_histogram()

        # Find the location of peaks in the symmetric_atn-delay plane
        self.peak_indices = self.find_peaks2(hist, self.peak_threshold,
                                             np.array([self.attenuation_min_distance, self.delay_min_distance]),
                                             self.num_sources)

        # compute delay_peak, attenuation peak, and attenuation/delay estimates
        delay_peak, atn_delay_est, atnpeak = self.convert_peaks(agrid, dgrid)
        
        # compute masks for separation
        bestind, mask = self.compute_masks(atnpeak, delay_peak)
       
        # demix with ML alignment and convert to time domain
        source_estimates = self.convert_time_domain(bestind, mask, atnpeak, delay_peak)

        self.separated_sources = source_estimates
        return source_estimates, atn_delay_est

    def compute_spectrogram(self, fs):
        """ Creates the stfts matrices for channel 0 and 1, and computes the frequency matrix.
        Parameter:
        fs (integer): sample rate

        Returns:
            stft_ch0 (np.matrix): a 2D Numpy matrix containing the stft of channel 0
            stft_ch1 (np.matrix): a 2D Numpy matrix containing the stft of channel 1
            wmat: a frequency matrix
        """
        if self.audio_signal.num_channels != 2:  # double check this
            raise ValueError('Cannot run Duet on audio signal without exactly 2 channels!')

        # Compute the stft of the two channel mixtures
        self.audio_signal.stft_params = self.stft_params
        self.audio_signal.stft()

        stft_ch0 = self.audio_signal.get_stft_channel(0)
        stft_ch1 = self.audio_signal.get_stft_channel(1)
        frequency_vector = self.audio_signal.freq_vector
        time_vector = self.audio_signal.time_bins_vector

        # remove dc component to avoid dividing by zero freq. in the delay estimation
        stft_ch0 = stft_ch0[1::, :]
        stft_ch1 = stft_ch1[1:, :]
        self.num_frequency_bins = len(frequency_vector)
        self.num_time_bins = len(time_vector)

        # Compute the freq. matrix for later use in phase calculations
        wmat = np.array(np.tile(np.mat(frequency_vector[1:]).T, (1, self.num_time_bins))) * (2 * np.pi / fs)  # WTF?

        return stft_ch0, stft_ch1, wmat

    def compute_atn_delay(self):
        # Calculate the symmetric attenuation (alpha) and delay (delta) for each
        # time-freq. point
        r21 = (self.stft_ch1 + constants.EPSILON) / (self.stft_ch0 + constants.EPSILON)
        atn = np.abs(r21)  # relative attenuation between the two channels
        symmetric_atn = atn - 1 / atn  # symmetric attenuation
        delay = -np.imag(np.log(r21)) / (2 * np.pi * self.wmat)  # relative delay
        return symmetric_atn, delay

    def make_histogram(self):
        """Receives the stft of the two channel mixtures and the frequency matrix to a create
        a smooth and normalized histogram.

        Parameters:
        stft_ch0: a 2D Numpy matrix containing the stft of channel 0
        stft_ch1: a 2D Numpy matrix containing the stft of channel 1
        symmetric_atn: the symmetric attenuation between two channels
        delay: the time delay between 2 channels
        wmat: a 2D Numpy matrix containing the time frequency matrix of the signal

        Returns:
            hist (np.array): a smooth and normaliz
            agrid:
            dgrid:
        """

        # What is going on here???
        # ------------------------------------
        # calculate the weighted histogram
        p = 1
        q = 0
        tfw = (np.abs(self.stft_ch0) * np.abs(self.stft_ch1)) ** p * (np.abs(self.wmat)) ** q  # time-freq weights

        # only consider time-freq. points yielding estimates in bounds
        a_premask = np.logical_and(self.attenuation_min < self.symmetric_atn, self.symmetric_atn < self.attenuation_max)
        d_premask = np.logical_and(self.delay_min < self.delay, self.delay < self.delay_max)
        ad_premask = np.logical_and(a_premask, d_premask)

        ad_nzind = np.nonzero(ad_premask)
        symmetric_atn_vec = self.symmetric_atn[ad_nzind]
        delay_vec = self.delay[ad_nzind]
        tfw_vec = tfw[ad_nzind]

        # compute the histogram
        H = np.histogram2d(symmetric_atn_vec, delay_vec, bins=np.array([self.num_attenuation_bins, self.num_delay_bins]),
                           range=np.array([[self.attenuation_min, self.attenuation_max],
                                           [self.delay_min, self.delay_max]]), normed=False,
                           weights=tfw_vec)

        hist = H[0] / H[0].max()
        agrid = H[1] # attenuation bins
        dgrid = H[2] # delay bins

        # Save these for later
        self.attenuation_grid = agrid
        self.delay_grid = dgrid
        self.attenuation_delay_histogram = hist
        self.non_normalized_hist = H[0]
        self.smoothed_hist = self._smooth_matrix(self.non_normalized_hist, np.array([3]))

        # smooth the histogram - local average 3-by-3 neighboring bins
        hist = self._smooth_matrix(hist, np.array([3]))

        # normalize and plot the histogram
        hist /= hist.max()

        return hist, agrid, dgrid 

    @staticmethod
    def find_peaks2(data, min_thr=0.5, min_dist=None, max_peaks=1):
        """Receives a matrix of positive numerical values (in [0,1]) and finds the peak values and corresponding 
        indices.

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

    def convert_peaks(self, agrid, dgrid):
        """Receives the attenuation and delay bins and computes the delay/attenuation peaks based on the peak finder indices.

       Parameters:
            agrid: Attenuation bins
            dgrid: Delay bins

        Returns:
            delay_peak:
            atn_delay_est:
            atnpeak:
        """
        symmetric_atn_peak = agrid[self.peak_indices[0, :]]
        delay_peak = dgrid[self.peak_indices[1, :]]

        atn_delay_est = np.vstack([symmetric_atn_peak, delay_peak]).T

        # convert symmetric_atn to atnpeak
        atnpeak = (symmetric_atn_peak + np.sqrt(symmetric_atn_peak ** 2 + 4)) / 2
        return delay_peak, atn_delay_est, atnpeak

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

    def compute_masks(self, atnpeak, delay_peak):
        # compute masks for separation
        bestsofar = np.inf * np.ones((self.num_frequency_bins - 1, self.num_time_bins))
        bestind = np.zeros((self.num_frequency_bins - 1, self.num_time_bins), int)
        for i in range(0, self.num_sources):
            score = np.abs(atnpeak[i] * np.exp(-1j * self.wmat * delay_peak[i]) * self.stft_ch0 - self.stft_ch1) ** 2 / \
                    (1 + atnpeak[i] ** 2)
            mask = (score < bestsofar)
            bestind[mask] = i
            bestsofar[mask] = score[mask]
        return bestind, mask

    def convert_time_domain(self, bestind, mask, atnpeak, delay_peak):
        source_estimates = np.zeros((self.num_sources, self.audio_signal.signal_length))
        for i in range(0, self.num_sources):
            mask = (bestind == i)
            Xm = np.vstack([np.zeros((1, self.num_time_bins)),
                            (self.stft_ch0 + atnpeak[i] * np.exp(1j * self.wmat * delay_peak[i]) * self.stft_ch1) /
                            (1 + atnpeak[i] ** 2) * mask])
            # xi = spectral_utils.f_istft(Xm, L, winType, hop, fs)
            xi = spectral_utils.e_istft(Xm, self.stft_params.window_length, 
                                        self.stft_params.hop_length, self.stft_params.window_type)

            source_estimates[i, :] = utils.add_mismatched_arrays(source_estimates[i, ], xi)[:self.audio_signal.signal_length]
            # add back to the separated signal a portion of the mixture to eliminate
            # most of the masking artifacts
            # source_estimates=source_estimates+0.05*x[0,:]
        return source_estimates

    @staticmethod
    def _smooth_matrix(matrix, kernel):
        """Performs two-dimensional convolution in order to smooth the values of matrix elements.

        (similar to low-pass filtering)

        Parameters:
            matrix (np.array): a 2D Numpy matrix to be smoothed
            kernel (np.array): a 2D Numpy matrix containing kernel values
        Note:
            if Kernel is of size 1 by 1 (scalar), a Kernel by Kernel matrix of 1/Kernel**2 will be used as the matrix
            averaging kernel
        Output:
            SMat (np.array): a 2D Numpy matrix containing a smoothed version of Mat (same size as Mat)
        """

        # check the dimensions of the Kernel matrix and set the values of the averaging
        # matrix, Kmat
        if np.prod(kernel.shape) == 1:
            Kmat = np.ones((kernel, kernel)) / kernel ** 2
        else:
            Kmat = kernel

        # make Kmat have odd dimensions
        krow, kcol = np.shape(Kmat)
        if np.mod(krow, 2) == 0:
            Kmat = signal.convolve2d(Kmat, np.ones((2, 1))) / 2
            krow += 1

        if np.mod(kcol, 2) == 0:
            Kmat = signal.convolve2d(Kmat, np.ones((1, 2))) / 2
            kcol += 1

        # adjust the matrix dimension for convolution
        copy_row = int(np.floor(krow / 2))  # number of rows to copy on top and bottom
        copy_col = int(np.floor(kcol / 2))  # number of columns to copy on either side

        # TODO: This is very ugly. Make this readable
        # form the augmented matrix (rows and columns added to top, bottom, and sides)
        matrix = np.mat(matrix)  # make sure Mat is a Numpy matrix
        augMat = np.vstack(
            [
                np.hstack(
                    [matrix[0, 0] * np.ones((copy_row, copy_col)),
                     np.ones((copy_row, 1)) * matrix[0, :],
                     matrix[0, -1] * np.ones((copy_row, copy_col))
                     ]),
                np.hstack(
                    [matrix[:, 0] * np.ones((1, copy_col)),
                     matrix,
                     matrix[:, -1] * np.ones((1, copy_col))]),
                np.hstack(
                    [matrix[-1, 1] * np.ones((copy_row, copy_col)),
                     np.ones((copy_row, 1)) * matrix[-1, :],
                     matrix[-1, -1] * np.ones((copy_row, copy_col))
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

        AA = np.tile(self.attenuation_grid[1::], (self.num_delay_bins, 1)).T
        DD = np.tile(self.delay_grid[1::].T, (self.num_attenuation_bins, 1))

        histogram_data = self.attenuation_delay_histogram if normalize else self.smoothed_hist

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
