#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.fftpack as scifft

import FftUtils
import SeparationBase
import Constants
import AudioSignal


class Repet(SeparationBase.SeparationBase):
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
            window_attributes (WindowAttributes): WindowAttributes object describing the window used in the repet
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
    def __init__(self, audio_signal, repet_type=None, window_attributes=None, sample_rate=None,
                 similarity_threshold=None, min_distance_between_frames=None, max_repeating_frames=None,
                 min_period=None, max_period=None, period=None, high_pass_cutoff=None):
        self.__dict__.update(locals())
        super(Repet, self).__init__(window_attributes=window_attributes, sample_rate=sample_rate,
                                    audio_signal=audio_signal)
        self.repet_type = RepetType.DEFAULT if repet_type is None else repet_type
        self.high_pass_cutoff = 100 if high_pass_cutoff is None else high_pass_cutoff

        if self.repet_type not in RepetType.all_types:
            raise TypeError('\'repet_type\' in Repet() constructor cannot be {}'.format(repet_type))

        if self.repet_type == RepetType.SIM:
            # if similarity_threshold == 0:
            #     raise Warning('Using default value of 1 for similarity_threshold.')
            # if min_distance_between_frames == 1:
            #     raise Warning('Using default value of 0 for min_distance_between_frames.')
            # if max_repeating_frames == 100:
            #     raise Warning('Using default value of 100 for maxRepeating frames.')

            self.similarity_threshold = 0 if similarity_threshold is None else similarity_threshold
            self.min_distance_between_frames = 1 if min_distance_between_frames is None else min_distance_between_frames
            self.max_repeating_frames = 10 if max_repeating_frames is None else max_repeating_frames
        elif self.repet_type == RepetType.ORIGINAL:

            if period is None:
                self.min_period = 0.8 if min_period is None else min_period
                self.max_period = min(8, self.audio_signal.signal_length / 3) if max_period is None else max_period
                self.min_period = self._update_period(self.min_period)
                self.max_period = self._update_period(self.max_period)
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
            repet = nussl.Repet(signal, window_type=nussl.RepetType.SIM, window_attributes=win)
            repet.min_distance_between_frames = 0.1
            repet.run()

        """

        # unpack window parameters
        win_len, win_type, win_ovp, nfft = self.window_attributes.window_length, self.window_attributes.window_type, \
                                           self.window_attributes.window_overlap_samples, self.window_attributes.num_fft

        # High pass filter cutoff freq. (in # of freq. bins)
        self.high_pass_cutoff = np.ceil(float(self.high_pass_cutoff) * (nfft - 1) / self.sample_rate)

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
        self.bkgd = np.zeros_like(self.audio_signal.audio_data)
        for i in range(N):
            RepMask = mask(self.real_spectrum[:, :, i], S)
            RepMask[1:self.high_pass_cutoff, :] = 1  # high-pass filter the foreground
            XMi = RepMask * self.complex_spectrum[:, :, i]
            yi = FftUtils.f_istft(XMi, win_len, win_type, win_ovp, self.sample_rate)[0]
            self.bkgd[i,] = yi[0:M]

        # self.bkgd = self.bkgd.T
        self.bkgd = AudioSignal.AudioSignal(audio_data_array=self.bkgd)

        return self.bkgd

    def _compute_spectrum(self):

        # compute the spectrograms of all channels
        N, M = self.audio_signal.audio_data.shape
        self.complex_spectrum = FftUtils.f_stft(self.audio_signal.get_channel(1),
                                                window_attributes=self.window_attributes, sample_rate=self.sample_rate)[0]

        for i in range(1, N):
            Sx = FftUtils.f_stft(self.audio_signal.get_channel(i), window_attributes=self.window_attributes,
                                 sample_rate=self.sample_rate)[0]
            self.complex_spectrum = np.dstack([self.complex_spectrum, Sx])

        self.real_spectrum = np.abs(self.complex_spectrum)
        if N == 1:
            self.complex_spectrum = self.complex_spectrum[:, :, np.newaxis]
            self.real_spectrum = self.real_spectrum[:, :, np.newaxis]

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
        V = np.mean(self.real_spectrum, axis=2)
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
        self.beat_spectrum = self.compute_beat_spectrum(np.mean(self.real_spectrum ** 2, axis=2))
        return self.beat_spectrum

    def _do_repet_sim(self):
        # unpack window overlap
        ovp = self.window_attributes.window_overlap_samples

        Vavg = np.mean(self.real_spectrum, axis=2)
        S = self.compute_similarity_matrix(Vavg)

        self.min_distance_between_frames = np.round([self.min_distance_between_frames * self.sample_rate / ovp])
        S = self.find_similarity_indices(S)

        return S

    def _do_repet_original(self):
        self.beat_spectrum = self.compute_beat_spectrum(np.mean(self.real_spectrum ** 2, axis=2))
        self.repeating_period = self.find_repeating_period(self.beat_spectrum, self.min_period, self.max_period)
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
            X[i, :] = Xi / (rowNorm + Constants.EPSILON)

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
        M = (W + Constants.EPSILON) / (V + Constants.EPSILON)

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
        X = np.hstack([X, np.zeros_like(X)])
        Sx = np.abs(scifft.fft(X, axis=1) ** 2)  # fft over columns (take the fft of each row at a time)
        Rx = np.real(scifft.ifft(Sx, axis=1)[:, 0:Lt])  # ifft over columns
        NormFactor = np.tile(np.arange(1, Lt + 1)[::-1], (Lf, 1))  # normalization factor
        Rx = Rx / NormFactor

        # compute the beat spectrum
        b = np.mean(Rx, axis=0)  # average over frequencies

        return b

    @staticmethod
    def find_repeating_period(beat_spectrum, min_period, max_period):
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
    def compute_repeating_mask_with_beat_spectrum(V, p):
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

        Wrow = W.flatten()  # np.reshape(W, (1, Lf * Lt))
        Vrow = V.flatten()  # np.reshape(V, (1, Lf * Lt))
        W = np.min(np.vstack([Wrow, Vrow]), axis=0)
        W = np.reshape(W, (Lf, Lt))
        M = (W + Constants.EPSILON) / (V + Constants.EPSILON)

        return M

    def _update_period(self, period):
        period = float(period)
        result = period * self.audio_signal.sample_rate
        result += self.window_attributes.window_length / self.window_attributes.window_overlap_samples - 1
        result /= self.window_attributes.window_overlap_samples
        return np.ceil(result)

    def plot(self, output_file, **kwargs):
        """ Creates a plot of either the beat spectrum or similarity matrix for this file and outputs to output_file.
            By default, the repet_type is used to determine which to plot,
            (original -> beat spectrum. sim -> similarity matrix)
            You can override this by passing in plot_beat_spectrum=True or plot_sim_matrix=True as parameters.
            You cannot set both of these overrides simultaneously.

        Parameters:
            output_file: string representing a path to the desired output file to be created.
            plot_beat_spectrum: Setting this will force plotting the beat spectrum
            plot_sim_matrix: Setting this will force plotting the similarity matrix

        EXAMPLE:
            To plot the beat spectrum you have a few options:
            1) (recommended)
            ::
            # set up AudioSignal
            signal = nussl.AudioSignal('path_to_file.wav')

            # by default, this Repet object is now set to the original repet (RepetType.ORIGINAL)
            repet = nussl.Repet(signal)

            # plots beat spectrum by default
            repet.plot('new_beat_spec_plot.png')

            2)
            ::
            # set up AudioSignal
            signal = nussl.AudioSignal('path_to_file.wav')

            # by giving this Repet object RepetType.SIM, it will default to printing the similarity matrix
            repet = nussl.Repet(signal, repet_type=nussl.RepetType.SIM)

            # but we can override this Repet object plotting the similarity matrix with this argument
            repet.plot('new_sim_matrix_plot.png', plot_beat_spectrum=True)

            To plot the similarity matrix you have a few options:
            1) (recommended)
            ::
            # set up AudioSignal
            signal = nussl.AudioSignal('path_to_file.wav')

            # by giving this Repet object RepetType.SIM, it will default to printing the similarity matrix
            repet = nussl.Repet(signal, repet_type=nussl.RepetType.SIM)

            # plots similarity matrix by default
            repet.plot('new_sim_matrix_plot.png')

            2)
            ::
            # set up AudioSignal
            signal = nussl.AudioSignal('path_to_file.wav')

            # by default, this Repet object is now set to the original repet (RepetType.ORIGINAL)
            repet = nussl.Repet(signal)

            # override plotting the beat spectrum with this argument
            repet.plot('new_sim_matrix_plot.png', plot_beat_spectrum=True)

            NOTE: You cannot do
            ::
            repet.plot('new_plot.png', plot_beat_spectrum=True, plot_sim_matrix=True)

            this will cause nussl to throw an exception!
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
            signal = nussl.
        """
        self.fgnd = self.audio_signal - self.bkgd
        return [self.bkgd, self.fgnd]


class RepetType():
    """Types of Repet algorithm implementation.
    """
    ORIGINAL = 'original'
    SIM = 'sim'
    ADAPTIVE = 'adaptive'
    DEFAULT = ORIGINAL
    all_types = [ORIGINAL, SIM, ADAPTIVE]

    def __init__(self):
        pass
