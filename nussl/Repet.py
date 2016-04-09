#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.fftpack as scifft
import scipy.spatial.distance

import spectral_utils
import separation_base
import constants
from audio_signal import AudioSignal


class Repet(separation_base.SeparationBase):
    """Implements the REpeating Pattern Extraction Technique algorithm using the Similarity Matrix (REPET-SIM).

    REPET is a simple method for separating the repeating background from the non-repeating foreground in a piece of
    audio mixture. REPET-SIM is a generalization of REPET, which looks for similarities instead of periodicities.

    References:

        * Zafar Rafii and Bryan Pardo. "Audio Separation System and Method," US20130064379 A1, US 13/612,413, March 14,
          2013
        * Zafar Rafii and Bryan Pardo. "Music/Voice Separation using the Similarity Matrix," 13th International Society
          on Music Information Retrieval, Porto, Portugal, October 8-12, 2012.

    See Also:
        http://music.eecs.northwestern.edu/research.php?project=repet

    Parameters:
            audio_signal (AudioSignal): audio mixture (M by N) containing M channels and N time samples
            repet_type (RepetType): Variant of Repet algorithm to perform.
            stft_params (WindowAttributes): WindowAttributes object describing the window used in the repet
             algorithm
            sample_rate (int): the sample rate of the audio signal
            similarity_threshold (Optional[int]): Used for RepetType.SIM. Defaults to 0
            min_distance_between_frames (Optional[int]): Used for RepetType.SIM. Defaults to 1
            max_repeating_frames (Optional[int]): Used for RepetType.SIM. Defaults to 10
            period (Optional[float]): Used for RepetType.ORIGINAL. The Period of the repeating part of the signal.
            min_period (Optional[float]): Used for RepetType.ORIGINAL. Only used if Period is not provided. Defaults to
             0.8
            max_period (Optional[float]): Used for RepetType.ORIGINAL. Only used if Period is not provided. Defaults to
             min(8, self.Mixture.SignalLength/3)
            high_pass_cutoff (Optional[int]): Defaults to 100

    Examples:
        :ref:`The REPET Demo Example <repet_demo>`
    """
    def __init__(self, input_audio_signal=None, sample_rate=None, stft_params=None, repet_type=None,
                 similarity_threshold=None, min_distance_between_frames=None, max_repeating_frames=None,
                 min_period=None, max_period=None, period=None, high_pass_cutoff=None):
        self.__dict__.update(locals())
        super(Repet, self).__init__(input_audio_signal=input_audio_signal,
                                    sample_rate=sample_rate, stft_params=stft_params)
        self.repet_type = RepetType.DEFAULT if repet_type is None else repet_type
        self.high_pass_cutoff = 100 if high_pass_cutoff is None else high_pass_cutoff

        if self.repet_type not in RepetType.all_types:
            raise TypeError('\'repet_type\' in Repet() constructor cannot be {}'.format(repet_type))

        if self.repet_type == RepetType.SIM:
            # if similarity_threshold == 0:
            #     warnings.warn('Using default value of 1 for similarity_threshold.')
            # if min_distance_between_frames == 1:
            #     warnings.warn('Using default value of 0 for min_distance_between_frames.')
            # if max_repeating_frames == 100:
            #     warnings.warn('Using default value of 100 for maxRepeating frames.')

            self.similarity_threshold = 0 if similarity_threshold is None else similarity_threshold
            self.min_distance_between_frames = 1 if min_distance_between_frames is None else min_distance_between_frames
            self.max_repeating_frames = 10 if max_repeating_frames is None else max_repeating_frames
        elif self.repet_type == RepetType.ORIGINAL:

            if period is None:
                self.min_period = 0.8 if min_period is None else min_period
                self.max_period = min(8, self.audio_signal.signal_duration / 3) if max_period is None else max_period
            else:
                self.period = period
                self.period = self._update_period(self.period)

        elif self.repet_type == RepetType.ADAPTIVE:
            raise NotImplementedError('Not allowed to do this yet!')

        self.verbose = False

    # @property
    def run(self):
        """Runs the REPET algorithm

        Returns:
            y (AudioSignal): repeating background (M by N) containing M channels and N time samples
            (the corresponding non-repeating foreground is equal to x-y)

        Example:
             ::
            signal = nussl.AudioSignal(path_to_input_file='input_name.wav')

            # Set up window parameters
            win = nussl.WindowAttributes(signal.sample_rate)
            win.window_length = 2048
            win.window_type = nussl.WindowType.HAMMING

            # Set up and run Repet
            repet = nussl.Repet(signal, window_type=nussl.RepetType.SIM, stft_params=win)
            repet.min_distance_between_frames = 0.1
            repet.run()

        """

        # unpack window parameters
        win_len, win_type, hop_len, nfft = self.stft_params.window_length, self.stft_params.window_type, \
                                           self.stft_params.hop_length, self.stft_params.n_fft_bins

        # High pass filter cutoff freq. (in # of freq. bins), +1 to match MATLAB implementation
        self.high_pass_cutoff = np.ceil(float(self.high_pass_cutoff) * (nfft - 1) / self.sample_rate) + 1

        self._compute_spectrum()

        # run the specific algorithm
        mask = None
        S = None
        if self.repet_type == RepetType.SIM:
            S = self._do_repet_sim()
            mask = self.compute_repeating_mask_sim

        elif self.repet_type == RepetType.ORIGINAL:
            S = self._do_repet_original()
            mask = self.compute_repeating_mask_with_beat_spectrum

        elif self.repet_type == RepetType.ADAPTIVE:
            raise NotImplementedError('How did you get into this state????')

        # separate the mixture background by masking
        N, M = self.audio_signal.audio_data.shape
        bkgd = np.zeros_like(self.audio_signal.audio_data)
        for i in range(N):
            repeating_mask = mask(self.magnitude_spectrogram[:, :, i], S)
            repeating_mask[1:self.high_pass_cutoff, :] = 1  # high-pass filter the foreground
            repeating_mask = np.vstack((repeating_mask, repeating_mask[-2:0:-1, :].conj()))
            stft_with_mask = repeating_mask * self.stft[:, :, i]
            y = spectral_utils.e_istft(stft_with_mask, win_len, hop_len, win_type,
                                       reconstruct_reflection=False, remove_padding=True)
            bkgd[i,] = y[0:M]

        self.bkgd = AudioSignal(audio_data_array=bkgd, sample_rate=self.sample_rate)

        return self.bkgd

    def _compute_spectrum(self):
        self.stft = self.audio_signal.stft(self.stft_params.window_length, self.stft_params.hop_length,
                                           self.stft_params.window_type, self.stft_params.n_fft_bins,
                                           overwrite=False, remove_reflection=False)

        self.magnitude_spectrogram = np.abs(self.stft[0:self.stft_params.window_length//2 + 1, :, :])

    def get_similarity_matrix(self):
        """Calculates and returns the similarity matrix for the audio file associated with this object

        Returns:
             similarity_matrix (np.array): similarity matrix for the audio file.

        EXAMPLE:
             ::
            # Set up audio signal
            signal = nussl.AudioSignal('path_to_file.wav')

            # Set up a Repet object
            repet = nussl.Repet(signal)

            # I don't have to run repet to get a similarity matrix for signal
            sim_mat = repet.get_similarity_matrix()

        """
        self._compute_spectrum()
        V = np.mean(self.magnitude_spectrogram, axis=2)
        self.similarity_matrix = self.compute_similarity_matrix(V)
        return self.similarity_matrix

    def get_beat_spectrum(self):
        """Calculates and returns the beat spectrum for the audio file associated with this object

        Returns:
            beat_spectrum (np.array): beat spectrum for the audio file

        EXAMPLE:
             ::
            # Set up audio signal
            signal = nussl.AudioSignal('path_to_file.wav')

            # Set up a Repet object
            repet = nussl.Repet(signal)

            # I don't have to run repet to get a beat spectrum for signal
            beat_spec = repet.get_beat_spectrum()
        """
        self._compute_spectrum()
        self.beat_spectrum = self.compute_beat_spectrum(np.mean(np.square(self.magnitude_spectrogram ** 2), axis=2))
        return self.beat_spectrum

    def _do_repet_sim(self):
        # unpack window overlap
        ovp = self.stft_params.window_overlap_samples

        Vavg = np.mean(self.magnitude_spectrogram, axis=2)
        S = self.compute_similarity_matrix(Vavg)

        self.min_distance_between_frames = np.round([self.min_distance_between_frames * self.sample_rate / ovp])
        S = self.find_similarity_indices(S)

        return S

    def _do_repet_original(self):
        self.min_period = self._update_period(self.min_period)
        self.max_period = self._update_period(self.max_period)
        self.beat_spectrum = self.compute_beat_spectrum(np.mean(np.square(self.magnitude_spectrogram), axis=2).T)
        self.repeating_period = self.find_repeating_period_simple(self.beat_spectrum, self.min_period, self.max_period)
        return self.repeating_period

    @staticmethod
    def compute_similarity_matrix(X):
        """Computes the similarity matrix using the cosine similarity for any given input matrix X.
        
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
            X[i, :] = Xi / (rowNorm + constants.EPSILON)

        # compute the similarity matrix    
        S = np.dot(X, X.T)
        return S

    def find_similarity_indices(self, S):
        """Finds the similarity indices for all time frames from the similarity matrix
        
        Parameters:
            S (np.array): similarity matrix (Lt by Lt)
                 
        Returns:
            I (np.array): similarity indices for all time frames
        """

        Lt = S.shape[0]
        I = np.zeros((Lt, self.max_repeating_frames))

        for i in range(0, Lt):
            pind = self.find_peaks(S[i, :], self.similarity_threshold,
                                   self.min_distance_between_frames, self.max_repeating_frames)
            I[i, :] = pind

        return I

    def find_peaks(self, data, min_thr=0.5, min_dist=None, max_num=1):
        """Finds the peak values and corresponding indices in a vector of real values in [0,1]
        
        Parameters:
            data (np.array): row vector of real values (in [0,1])
            min_thr: (Optional[float]) minimum threshold (in [0,1]) on data values. Defaults to 0.5
            min_dist:(Optional[int]) minimum distance (in # of elements) between peaks. Defaults to .25 * data.length
            max_num: (Optional[int]) maximum number of peaks. Defaults to 1
        
        Returns:
            peak_indices (np.array): sorted array of indices of peaks in the data
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
    def compute_repeating_mask_sim(V, I):
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
        M = (W + constants.EPSILON) / (V + constants.EPSILON)

        return M

    @staticmethod
    def compute_beat_spectrum(X):
        """Computes the beat spectrum averages (over freq.s) the autocorrelation matrix of a one-sided spectrogram.

         The autocorrelation matrix is computed by taking the autocorrelation of each row of the spectrogram and
         dismissing the symmetric half.

        Parameters:
            X (np.array): 2D matrix containing the one-sided power
            spectrogram of the audio signal (Lf by Lt by num channels)
        Returns:
            b (np.array): array containing the beat spectrum based on the power spectrogram
        """
        # compute the row-wise autocorrelation of the input spectrogram
        Lf, Lt = X.shape
        X = np.vstack([X, np.zeros_like(X)])
        Fx = scifft.fft(X, axis=0)
        Ax = np.abs(Fx)
        Sx = Ax ** 2  # fft over columns (take the fft of each row at a time)
        Rx = np.real(scifft.ifft(Sx, axis=0)[0:Lf,:])  # ifft over columns
        NormFactor = np.tile(np.arange(Lf, 0, -1), (Lt, 1)).T  # normalization factor
        Rx = Rx / NormFactor

        # compute the beat spectrum
        b = np.mean(Rx, axis=1)  # average over frequencies

        return b

    @staticmethod
    def find_repeating_period_simple(beat_spectrum, min_period, max_period):
        """Computes the repeating period of the sound signal using the beat spectrum using simple algorithm.

        Parameters:
            beat_spectrum (np.array): input beat spectrum array
            min_period (int): minimum possible period value
            max_period (int): maximum possible period value
        Returns:
             period (int) : The period of the sound signal in stft time bins
        """

        beat_spectrum = beat_spectrum[1:]  # discard the first element of beat_spectrum (lag 0)
        beat_spectrum = beat_spectrum[min_period - 1:  max_period]
        period = np.argmax(beat_spectrum) + min_period

        return period

    @staticmethod
    def find_repeating_period_complex(beat_spectrum):
        auto_cosine = np.zeros((len(beat_spectrum), 1))

        for i in range(0, len(beat_spectrum) - 1):
            auto_cosine[i] = 1 - scipy.spatial.distance.cosine(beat_spectrum[0:len(beat_spectrum) - i],
                                                               beat_spectrum[i:len(beat_spectrum)])

        ac = auto_cosine[0:np.floor(auto_cosine.shape[0])/2]
        auto_cosine = np.vstack([ac[1], ac, ac[-2]])
        auto_cosine_diff = np.ediff1d(auto_cosine)
        sign_changes = auto_cosine_diff[0:-1]*auto_cosine_diff[1:]
        sign_changes = np.where(sign_changes < 0)[0]

        extrema_values = ac[sign_changes]

        e1 = np.insert(extrema_values, 0, extrema_values[0])
        e2 = np.insert(extrema_values, -1, extrema_values[-1])

        extrema_neighbors = np.stack((e1[0:-1], e2[1:]))

        m = np.amax(extrema_neighbors, axis=0)
        extrema_values = extrema_values.flatten()
        maxima = np.where(extrema_values >= m)[0]
        maxima = zip(sign_changes[maxima], extrema_values[maxima])
        maxima = maxima[1:]
        maxima = sorted(maxima, key = lambda x: -x[1])
        period = maxima[0][0]

        return period

    @staticmethod
    def compute_repeating_mask_with_beat_spectrum(V, p):
        """Computes the soft mask for the repeating part using the magnitude spectrogram and the repeating period

        Parameters:
            V (np.array): 2D matrix containing the magnitude spectrogram of a signal (Lf by Lt)
            p (int): repeating period measured in # of time frames
        Returns:
            M (np.array): 2D matrix (Lf by Lt) containing the soft mask for the repeating part, elements of M take on
            values in [0,1]

        """
        p += 1 # this is a kluge to make this implementation match the original MATLAB implementation
        n, m = V.shape
        r = np.ceil(float(m) / p)
        W = np.hstack([V, float('nan') * np.zeros((n, r * p - m))])
        W = np.reshape(W.T, (r, n * p))
        W1 = np.median(W[0:r, 0:n * (m - (r - 1) * p)], axis=0)
        W2 = np.median(W[0:r - 1, n * (m - (r - 1) * p):n * p], axis=0)
        W = np.hstack([W1, W2])
        W = np.reshape(np.tile(W, (r, 1)), (r * p, n)).T
        W = W[:, 0:m]

        Wrow = W.flatten()  # np.reshape(W, (1, Lf * Lt))
        Vrow = V.flatten()  # np.reshape(V, (1, Lf * Lt))
        W = np.min(np.vstack([Wrow, Vrow]), axis=0)
        W = np.reshape(W, (n, m))
        M = (W + constants.EPSILON) / (V + constants.EPSILON)

        return M

    def update_periods(self):
        """
        Will update periods for use with self.find_repeating_period_simple(). Updates from seconds to stft bin values.
        Call this if you haven't done self.run() or else you won't get good results
        Examples:
            ::
            a = nussl.AudioSignal('path/to/file.wav')
            r = nussl.Repet(a)

            beat_spectrum = r.get_beat_spectrum()
            r.update_periods()
            repeating_period = r.find_repeating_period_simple(beat_spectrum, r.min_period, r.max_period)

        """
        self.period = self._update_period(self.period) if self.period is not None else None
        self.min_period = self._update_period(self.min_period) if self.min_period is not None else None
        self.max_period = self._update_period(self.max_period) if self.max_period is not None else None

    def _update_period(self, period):
        period = float(period)
        result = period * self.audio_signal.sample_rate
        result += self.stft_params.window_length / self.stft_params.window_overlap - 1
        result /= self.stft_params.window_overlap
        return np.ceil(result)

    def plot(self, output_file, **kwargs):
        """ Creates a plot of either the beat spectrum or similarity matrix for this file and outputs to output_file.
            By default, the repet_type is used to determine which to plot,
            (original -> beat spectrum. sim -> similarity matrix)
            You can override this by passing in plot_beat_spectrum=True or plot_sim_matrix=True as parameters.
            You cannot set both of these overrides simultaneously.

        Parameters:
            output_file (string) : string representing a path to the desired output file to be created.
            plot_beat_spectrum (Optional[bool]) : Setting this will force plotting the beat spectrum
            plot_sim_matrix (Optional[bool]) : Setting this will force plotting the similarity matrix

        EXAMPLE:
             ::
            # To plot the beat spectrum you have a few options:

            # 1) (recommended)
            # set up AudioSignal
            signal = nussl.AudioSignal('path_to_file.wav')

            # by default, this Repet object is now set to the original repet (RepetType.ORIGINAL)
            repet1 = nussl.Repet(signal)

            # plots beat spectrum by default
            repet1.plot('new_beat_spec_plot.png')

            # 2)
            # by giving this Repet object RepetType.SIM, it will default to plotting the similarity matrix
            repet2 = nussl.Repet(signal, repet_type=nussl.RepetType.SIM)

            # but we can override this Repet object plotting the similarity matrix with this argument
            repet2.plot('new_sim_matrix_plot.png', plot_beat_spectrum=True)

            # To plot the similarity matrix you have a few options:
            # 3) (recommended)
            # set up AudioSignal
            signal = nussl.AudioSignal('path_to_file.wav')

            # by giving this Repet object RepetType.SIM, it will default to printing the similarity matrix
            repet3 = nussl.Repet(signal, repet_type=nussl.RepetType.SIM)

            # plots similarity matrix by default
            repet3.plot('new_sim_matrix_plot.png')

            # 4)
            # by default, this Repet object is now set to the original repet (RepetType.ORIGINAL)
            repet4 = nussl.Repet(signal)

            # BUT we can override plotting the beat spectrum with this argument
            repet4.plot('new_sim_matrix_plot.png', plot_sim_matrix=True)

            # NOTE: You cannot do
            # repet.plot('new_plot.png', plot_beat_spectrum=True, plot_sim_matrix=True)
            # this will cause nussl to throw an exception!
        """
        import matplotlib.pyplot as plt
        plt.close('all')

        plot_beat_spectrum = self.repet_type is RepetType.ORIGINAL
        plot_sim_matrix = self.repet_type is RepetType.SIM

        if kwargs is not None:
            if kwargs.has_key('plot_beat_spectrum'):
                plot_beat_spectrum = kwargs['plot_beat_spectrum']
            if kwargs.has_key('plt_sim_matrix'):
                plot_sim_matrix = kwargs['plot_sim_matrix']

        if plot_beat_spectrum == plot_sim_matrix == True:
            raise AssertionError('Cannot set both plot_beat_spectrum=True and plot_sim_matrix=True!')

        if plot_beat_spectrum:
            plt.plot(self.get_beat_spectrum())
            plt.title('Beat Spectrum for {}'.format(self.audio_signal.file_name))
            plt.grid('on')

        elif plot_sim_matrix:
            plt.pcolormesh(self.get_similarity_matrix())
            plt.title('Similarity Matrix for {}'.format(self.audio_signal.file_name))

        plt.axis('tight')
        plt.savefig(output_file)

    def make_audio_signals(self):
        """ Returns the background and foreground audio signals

        Returns:
            Audio Signals (List): 2 element list.

                * bkgd: Audio signal with the calculated background track
                * fkgd: Audio signal with the calculated foreground track

        EXAMPLE:
             ::
            # set up AudioSignal object
            signal = nussl.AudioSignal('path_to_file.wav')

            # set up and run repet
            repet = nussl.Repet(signal)
            repet.run()

            # get audio signals (AudioSignal objects)
            background, foreground = repet.make_audio_signals()
        """
        self.fgnd = self.audio_signal - self.bkgd
        return [self.bkgd, self.fgnd]


class RepetType():
    """Types of Repet algorithm implementation.
    """
    ORIGINAL = 'original'
    SIM = 'sim'
    # ADAPTIVE = 'adaptive'
    DEFAULT = ORIGINAL
    all_types = [ORIGINAL, SIM]

    def __init__(self):
        pass
