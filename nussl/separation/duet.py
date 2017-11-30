#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
An implementation of the Degenerate Unmixing Estimation Technique (DUET) algorithm.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import mask_separation_base
import masks
from ..core import utils
from ..core import constants
from ..core.audio_signal import AudioSignal


class Duet(mask_separation_base.MaskSeparationBase):
    """
    The DUET algorithm was originally proposed by S.Rickard and F.Dietrich for DOA estimation
    and further developed for BSS and demixing by A.Jourjine, S.Rickard, and O. Yilmaz. DUET extracts
    sources using the symmetric attenuation and relative delay between two channels. The symmetric
    attenuation is calculated from the ratio of the two channels' stft amplitudes, and the delay is the arrival delay
    between the two sensors used to record the audio signal. These two values are clustered as peaks on a
    histogram to determine where each source occurs. This implementation of DUET creates and returns Mask objects after
    the run() function, which can then be applied to the original audio signal to extract each individual source.

    References:

        * Rickard, Scott. "The DUET blind source separation algorithm." Blind Speech Separation. Springer Netherlands,
          2007. 217-241.
        * Yilmaz, Ozgur, and Scott Rickard. "Blind separation of speech mixtures via time-frequency masking." Signal
          Processing, IEEE transactions on 52.7 (2004): 1830-1847.

    Parameters:
        input_audio_signal (np.array): a 2-row Numpy matrix containing samples of the two-channel mixture.
        num_sources (int): Number of sources to find.
        attenuation_min (int): Minimum distance in utils.find_peak_indices, change if not enough peaks are identified.
        attenuation_max (int): Used for creating a histogram without outliers.
        num_attenuation_bins (int): Number of bins for attenuation. 
        delay_min (int): Lower bound on delay, used as minimum distance in utils.find_peak_indices.
        delay_max (int): Upper bound on delay, used for creating a histogram without outliers.
        num_delay_bins (int): Number of bins for delay. 
        peak_threshold (float): Value in [0, 1] for peak picking. 
        attenuation_min_distance (int): Minimum distance between peaks wrt attenuation. 
        delay_min_distance (int): Minimum distance between peaks wrt delay.
        p (int): Weight the histogram with the symmetric attenuation estimator.
        q (int): Weight the histogram with the delay estimator
        On page 8 of his paper, Rickard recommends p=1 and q=0 as a default starting point and p=.5, q=0 if one
        source is more dominant.

    Attributes:
        stft_ch0 (np.array): A Numpy matrix containing the stft data of channel 0.
        stft_ch1 (np.array): A Numpy matrix containing the stft data of channel 1.
        frequency_matrix (np.array): A Numpy matrix containing the frequencies of analysis.
        symmetric_atn (np.array): A Numpy matrix containing the symmetric attenuation between the two channels.
        delay (np.array): A Numpy matrix containing the delay between the two channels.
        num_time_bins (np.array): The number of time bins for the frequency matrix and mask arrays.
        num_frequency_bins (int): The number of frequency bins for the mask arrays.
        attenuation_bins (int): A Numpy array containing the attenuation bins for the histogram.
        delay_bins (np.array): A Numpy array containing the delay bins for the histogram.
        normalized_attenuation_delay_histogram (np.array): A normalized Numpy matrix containing the
                                                           attenuation delay histogram, which has peaks for each source.
        attenuation_delay_histogram (np.array): A non-normalized Numpy matrix containing the
                                                attenuation delay histogram, which has peaks for each source.
        peak_indices (np.array): A Numpy array containing the indices of the peaks for the histogram.
        separated_sources (np.array): A Numpy array of arrays containing each separated source.

    Examples:
        :ref:`The DUET Demo Example <duet_demo>`

    """

    def __init__(self, input_audio_signal, num_sources,
                 attenuation_min=-3, attenuation_max=3, num_attenuation_bins=50,
                 delay_min=-3, delay_max=3, num_delay_bins=50,
                 peak_threshold=0.2, attenuation_min_distance=5, delay_min_distance=5, p=1, q=0):
        super(Duet, self).__init__(input_audio_signal=input_audio_signal,
                                   mask_type=mask_separation_base.MaskSeparationBase.BINARY_MASK)

        if not self.audio_signal.is_stereo:
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
        self.p = p
        self.q = q
        self.final_sources = []
        self.result_masks = []

        self.stft_ch0 = None
        self.stft_ch1 = None
        self.frequency_matrix = None
        self.symmetric_atn = None
        self.delay = None
        self.num_time_bins = None
        self.num_frequency_bins = None
        self.attenuation_bins = None
        self.delay_bins = None
        self.normalized_attenuation_delay_histogram = None
        self.attenuation_delay_histogram = None
        self.peak_indices = None
        self.delay_peak = None
        self.atn_peak = None
        self.separated_sources = None

    def run(self):
        """ Extracts N sources from a given stereo audio mixture (N sources captured via 2 sensors)

        Returns:
            computed_masks (np.array): A list of binary mask objects that can be used to extract the sources

        Example:
        .. code-block:: python
            :linenos:

            #Import input audio signal
            input_file_name = '../Input/dev1_female3_inst_mix.wav'
            signal = AudioSignal(path_to_input_file=input_file_name)

            # Set up and run Duet
            duet = Duet(signal, a_min=-3, a_max=3, a_num=50, d_min=-3, d_max=3, d_num=50, threshold=0.2,
            a_min_distance=5, d_min_distance=5, num_sources=3)
            duet.run()

            # plot histogram results
            duet.plot(os.path.join('..', 'Output', 'duet_2d.png'))
            duet.plot(os.path.join('..', 'Output', 'duet_3d.png'), three_d_plot=True)

            # Create output file for each source found
            output_name_stem = os.path.join('..', 'Output', 'duet_source')
            i = 1
            for s in duet.make_audio_signals():
                output_file_name = output_name_stem + str(i) + '.wav'
                s.write_audio_to_file(output_file_name)
                i += 1

        """
        self.result_masks = []

        if not self.audio_signal.is_stereo:  # double check this
            raise ValueError('Cannot run Duet on audio signal without exactly 2 channels!')

        # Calculate the stft of both channels and create the frequency matrix (the matrix containing the
        #  frequencies of analysis of the Fourier transform)
        self.stft_ch0, self.stft_ch1, self.frequency_matrix = self._compute_spectrogram(self.sample_rate)

        # Calculate the symmetric attenuation (alpha) and delay (delta) for each
        # time-freq. point and return a matrix for each
        self.symmetric_atn, self.delay = self._compute_atn_delay(self.stft_ch0, self.stft_ch1, self.frequency_matrix)

        # Make histogram of attenuation-delay values and get the center values for the bins in this histogram
        self.normalized_attenuation_delay_histogram, self.attenuation_bins, self.delay_bins = self._make_histogram()

        # Find the location of peaks in the attenuation-delay plane
        self.peak_indices = utils.find_peak_indices(self.normalized_attenuation_delay_histogram, self.num_sources,
                                                    threshold=self.peak_threshold,
                                                    min_dist=[self.attenuation_min_distance,
                                                              self.delay_min_distance])

        # compute delay_peak, attenuation peak, and attenuation/delay estimates
        self.delay_peak, atn_delay_est, self.atn_peak = self._convert_peaks(self.peak_indices)

        # compute masks for separation
        computed_masks = self._compute_masks()

        return computed_masks

    def get_atn_delay_histogram(self, recompute=False, normalized=False):
        """
        Computes the 2D attenuation/delay histogram for the given input signal.
        Args:
            recompute (bool): If `True`, this will recompute everything required to calculate the histogram
            (i.e., spectrograms). If `False`, then this will use previously calculated values if available.
            normalized (bool): Will normalize the histogram

        Returns:

        """

        if not self.audio_signal.is_stereo:  # double check this
            raise ValueError('Cannot run Duet on audio signal without exactly 2 channels!')

        if not np.all([self.stft_ch0, self.stft_ch1, self.frequency_matrix]) or recompute:
            self.stft_ch0, self.stft_ch1, self.frequency_matrix = self._compute_spectrogram(self.sample_rate)

        if not np.all([self.symmetric_atn, self.delay]) or recompute:
            self.symmetric_atn, self.delay = self._compute_atn_delay(self.stft_ch0, self.stft_ch1,
                                                                     self.frequency_matrix)

        if self.normalized_attenuation_delay_histogram is None or recompute:
            self.normalized_attenuation_delay_histogram, self.attenuation_bins, self.delay_bins = self._make_histogram()

        if normalized:
            return self.normalized_attenuation_delay_histogram
        else:
            return self.attenuation_delay_histogram

    def convert_peaks_to_masks(self, peak_indices=None):
        """

        Args:
            peak_indices:

        Returns:

        """
        if peak_indices is None and self.peak_indices is None:
            raise ValueError('Need peak_indices to make masks!')

        if self.normalized_attenuation_delay_histogram is None:
            self.get_atn_delay_histogram(recompute=True)

        peak_indices = self.peak_indices if peak_indices is None else peak_indices
        self.delay_peak, atn_delay_est, self.atn_peak = self._convert_peaks(peak_indices)
        return self._compute_masks()

    def _compute_spectrogram(self, sample_rate):
        """ Creates the STFT matrices for channel 0 and 1, and computes the frequency matrix.
        Parameter:
            sample_rate (integer): sample rate

        Returns:
            stft_ch0 (np.matrix): a 2D Numpy matrix containing the stft of channel 0
            stft_ch1 (np.matrix): a 2D Numpy matrix containing the stft of channel 1
            wmat (np.matrix): a 2D Numpy matrix containing the frequencies of analysis of the Fourier transform
        """

        # Compute the stft of the two channel mixtures
        self.audio_signal.stft_params = self.stft_params
        self.audio_signal.stft()

        stft_ch0 = self.audio_signal.get_stft_channel(0)
        stft_ch1 = self.audio_signal.get_stft_channel(1)

        # Compute the freq. matrix for later use in phase calculations
        n_time_bins = len(self.audio_signal.time_bins_vector)
        wmat = np.array(np.tile(np.mat(self.audio_signal.freq_vector).T, (1, n_time_bins))) * (2 * np.pi / sample_rate)
        wmat += constants.EPSILON
        return stft_ch0, stft_ch1, wmat

    @staticmethod
    def _compute_atn_delay(stft_ch0, stft_ch1, frequency_matrix):
        # Calculate the symmetric attenuation (alpha) and delay (delta) for each
        # time-freq. point
        inter_channel_ratio = (stft_ch1 + constants.EPSILON) / (stft_ch0 + constants.EPSILON)
        attenuation = np.abs(inter_channel_ratio)  # relative attenuation between the two channels
        symmetric_attenuation = attenuation - 1 / attenuation  # symmetric attenuation
        relative_delay = -np.imag(np.log(inter_channel_ratio)) / (2 * np.pi * frequency_matrix)  # relative delay
        return symmetric_attenuation, relative_delay

    def _make_histogram(self):
        """Receives the stft of the two channel mixtures and the frequency matrix to a create
        a smooth and normalized histogram.

        Parameters:
        stft_ch0 (complex np.array): a 2D Numpy matrix containing the stft of channel 0
        stft_ch1 (complex np.array): a 2D Numpy matrix containing the stft of channel 1
        symmetric_atn (np.array): the symmetric attenuation between two channels
        delay (np.array): the time delay between 2 channels
        wmat(np.array): a 2D Numpy matrix containing the frequency matrix of the signal

        Returns:
            histogram (np.array): a smooth and normalized histogram
            atn_bins (np.array): The range of attenuation values distributed into bins
            delay_bins (np.array): The range of delay values distributed into bins
        """
        # calculate the weighted histogram
        time_frequency_weights = (np.abs(self.stft_ch0) * np.abs(self.stft_ch1)) ** self.p * \
                                 (np.abs(self.frequency_matrix)) ** self.q

        # only consider time-freq. points yielding estimates in bounds
        attenuation_premask = np.logical_and(self.attenuation_min < self.symmetric_atn,
                                             self.symmetric_atn < self.attenuation_max)

        delay_premask = np.logical_and(self.delay_min < self.delay, self.delay < self.delay_max)
        attenuation_delay_premask = np.logical_and(attenuation_premask, delay_premask)

        nonzero_premask = np.nonzero(attenuation_delay_premask)
        symmetric_attenuation_vector = self.symmetric_atn[nonzero_premask]
        delay_vector = self.delay[nonzero_premask]
        time_frequency_weights_vector = time_frequency_weights[nonzero_premask]

        bins_array = np.array([self.num_attenuation_bins, self.num_delay_bins])
        range_array = np.array([[self.attenuation_min, self.attenuation_max], [self.delay_min, self.delay_max]])

        # compute the histogram
        histogram, atn_bins, delay_bins = np.histogram2d(symmetric_attenuation_vector, delay_vector,
                                                         bins=bins_array, range=range_array,
                                                         weights=time_frequency_weights_vector)

        # Save non-normalized as an option for plotting later
        self.attenuation_delay_histogram = histogram

        # Scale histogram from 0 to 1
        histogram /= histogram.max()

        # smooth the normalized histogram - local average 3-by-3 neighboring bins
        histogram = self._smooth_matrix(histogram, np.array([3]))
        return histogram, atn_bins, delay_bins

    def _convert_peaks(self, peak_indices):
        """Receives the attenuation and delay bins and computes the delay/attenuation
        peaks based on the peak finder indices.

        Returns:
            delay_peak(np.array): The delay peaks determined from the histogram
            atn_delay_est (np.array): The estimated symmetric attenuation and delay values
            atn_peak (np.array): Attenuation converted from symmetric attenuation
        """

        if peak_indices is None:
            raise ValueError('No peak indices')

        atn_indices = [x[0] for x in peak_indices]
        delay_indices = [x[1] for x in peak_indices]

        if any(a > len(self.attenuation_bins) for a in atn_indices):
            raise ValueError('Attenuation index greater than length of bins')

        symmetric_atn_peak = self.attenuation_bins[atn_indices]
        delay_peak = self.delay_bins[delay_indices]

        atn_delay_est = np.column_stack((symmetric_atn_peak, delay_peak))

        # convert symmetric_atn to atn_peak using formula from Rickard
        atn_peak = (symmetric_atn_peak + np.sqrt(symmetric_atn_peak ** 2 + 4)) / 2
        return delay_peak, atn_delay_est, atn_peak

    def _compute_masks(self):
        """Receives the attenuation and delay peaks and computes a mask to be applied to the signal for source
        separation.

        """
        # compute masks for separation
        best_so_far = np.inf * np.ones_like(self.stft_ch0, dtype=float)

        for i in range(0, self.num_sources):
            mask_array = np.zeros_like(self.stft_ch0, dtype=bool)
            phase = np.exp(-1j * self.frequency_matrix * self.delay_peak[i])
            score = np.abs(self.atn_peak[i] * phase * self.stft_ch0 - self.stft_ch1) ** 2 / (1 + self.atn_peak[i] ** 2)
            mask = (score < best_so_far)
            mask_array[mask] = True
            background_mask = masks.BinaryMask(np.array(mask_array))
            self.result_masks.append(background_mask)
            self.result_masks[0].mask = np.logical_xor(self.result_masks[i].mask, self.result_masks[0].mask)
            best_so_far[mask] = score[mask]

        # Compute first mask based on what the other masks left remaining
        self.result_masks[0].mask = np.logical_not(self.result_masks[0].mask)
        return self.result_masks

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
            smoothed_matrix (np.array): a 2D Numpy matrix containing a smoothed version of Mat (same size as Mat)
        """

        # check the dimensions of the Kernel matrix and set the values of the averaging
        # matrix, kernel_matrix
        if np.prod(kernel.shape) == 1:
            kernel_matrix = np.ones((kernel[0], kernel[0])) / kernel[0] ** 2
        else:
            kernel_matrix = kernel

        # make kernel_matrix have odd dimensions
        krow, kcol = np.shape(kernel_matrix)
        if np.mod(krow, 2) == 0:
            kernel_matrix = signal.convolve2d(kernel_matrix, np.ones((2, 1))) / 2
            krow += 1

        if np.mod(kcol, 2) == 0:
            kernel_matrix = signal.convolve2d(kernel_matrix, np.ones((1, 2))) / 2
            kcol += 1

        # adjust the matrix dimension for convolution
        copy_row = int(np.floor(krow / 2))  # number of rows to copy on top and bottom
        copy_col = int(np.floor(kcol / 2))  # number of columns to copy on either side

        # TODO: This is very ugly. Make this readable.
        # form the augmented matrix (rows and columns added to top, bottom, and sides)
        matrix = np.mat(matrix)  # make sure Mat is a Numpy matrix
        augmented_matrix = np.vstack(
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
        smooted_matrix = signal.convolve2d(augmented_matrix, kernel_matrix[::-1, ::-1], mode='valid')

        return smooted_matrix

    def make_audio_signals(self):
        """Returns the extracted signals

        Returns:
            signals (List[AudioSignal]): List of AudioSignals extracted using DUET.

        """
        if len(self.result_masks) == 0:
            raise ValueError('Cannot make audio signals with no masks to apply!')

        signals = []
        for i in range(self.num_sources):
            # Apply masks to stft channels using equation 8.34 (pg. 11) provided by Rickard
            stft_sources = ((self.stft_ch0 + self.atn_peak[i] * np.exp(1j * self.frequency_matrix * self.delay_peak[i])
                             * self.stft_ch1) / (1 + self.atn_peak[i] ** 2))
            new_sig = self.audio_signal.make_copy_with_stft_data(stft_sources, verbose=False)
            new_sig = new_sig.apply_mask(self.result_masks[i])
            new_sig.stft_params = self.stft_params
            source_estimate = new_sig.istft(overwrite=True, truncate_to_length=self.audio_signal.signal_length)
            cur_signal = AudioSignal(audio_data_array=source_estimate, sample_rate=self.sample_rate)
            signals.append(cur_signal)
        return signals

    def plot(self, output_name, three_d_plot=False, normalize=True):
        """Plots histograms with the results of the DUET algorithm

        Parameters:
            output_name (str): path to save plot as
            three_d_plot (Optional[bool]): Flags whether or not to plot in 3d. Defaults to False
            normalize (Optional[bool]): Flags whether the matrix should be normalized or not
        """
        plt.close('all')

        histogram_data = self.get_atn_delay_histogram(recompute=True, normalized=normalize) \
            if self.attenuation_delay_histogram is None \
            else self.attenuation_delay_histogram

        atn_tile = np.tile(self.attenuation_bins[1:], (self.num_delay_bins, 1)).T
        delay_tile = np.tile(self.delay_bins[1:].T, (self.num_attenuation_bins, 1))

        # plot the histogram in 2D
        if not three_d_plot:
            plt.figure()
            plt.pcolormesh(atn_tile, delay_tile, histogram_data)
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
            ax.plot_wireframe(atn_tile, delay_tile, histogram_data, rstride=2, cstride=2)
            plt.xlabel(r'$\alpha$', fontsize=16)
            plt.ylabel(r'$\delta$', fontsize=16)
            plt.title(r'$\alpha-\delta$ Histogram')
            plt.axis('tight')
            ax.view_init(30, 30)
            plt.savefig(output_name)
            plt.close()
