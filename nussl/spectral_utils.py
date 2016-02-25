#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as spfft
from scipy.signal import hamming, hann, blackman

import Constants


def f_stft(signal, num_ffts, win_length, window_type, window_overlap, sample_rate):
    """Computes the one-sided STFT of a signal

    Parameters:
        signal (np.array): row vector containing the signal.
        num_ffts [int]: min number of desired freq. samples in (-pi,pi]. MUST be >= L. Defaults to
         int(2 ** np.ceil(np.log2(win_length)))
        stft_params ([WindowAttributes]): Contains all info about windowing for stft.
        win_length (Optional[int]): length of one window (in # of samples)
        window_type (Optional[WindowType]): window type
        window_overlap (Optional[int]): number of overlapping samples between adjacent windows
        sample_rate (int): sampling rate of the signal

    Note:
        Either stft_params or all of [win_length, window_type, window_overlap, and num_ffts] must be provided.

    Returns:
        * **S** (*np.array*) - 2D numpy matrix containing the one-sided short-time Fourier transform of the signal
         (complex)
        * **P** (*np.array*) - 2D numpy matrix containing the one-sided PSD of the signal
        * **F** (*np.array*) - frequency vector
        * **T** (*np.array*) - time vector
    """
    if num_ffts is None:
        num_ffts = int(2 ** np.ceil(np.log2(win_length)))

    signal = np.mat(signal)

    # split data into blocks (make sure X is a row vector)
    if signal.shape[0] != 1:  # TODO: X.ndim? need a better check here
        raise ValueError('X must be a row vector')
    elif num_ffts < win_length:
        raise ValueError('num_fft_bins must be greater or equal the window length (L)!')

    Hop = int(win_length - window_overlap)
    N = signal.shape[1]

    # zero-pad the vector at the beginning and end to reduce the window tapering effect
    if np.mod(win_length, 2) == 0:
        zp1 = win_length / 2
    else:
        zp1 = (win_length - 1) / 2

    signal = np.hstack([np.zeros((1, zp1)), signal, np.zeros((1, zp1))])
    N += 2 * zp1

    # zero pad if N-2*zp1 is not an integer multiple of Hop
    rr = np.mod(N - 2 * zp1, Hop)
    if rr != 0:
        zp2 = Hop - rr
        signal = np.hstack([signal, np.zeros((1, zp2))])
        N = signal.shape[1]
    else:
        zp2 = 0

    NumBlock = int(((N - win_length) / Hop) + 1)

    window = make_window(window_type, win_length)
    Wnorm2 = np.dot(window, window.T)

    # Generate freq. vector
    freq = (sample_rate / 2) * np.linspace(0, 1, num=num_ffts / 2 + 1)  # Frequency Bins?
    lenFreq = len(freq)

    # Take the fft of each block
    S = 1j * np.zeros((NumBlock, lenFreq))  # row: time, col: freq. to increase speed
    P = np.zeros((NumBlock, lenFreq))

    for i in range(0, NumBlock):
        Xw = np.multiply(window, signal[0, (i * Hop):(i * Hop + win_length)])
        XX = spfft.fft(Xw, n=num_ffts)
        XX_trun = XX[0, 0:lenFreq]

        S[i, :] = XX_trun
        P[i, :] = (1 / float(sample_rate)) * ((abs(S[i, :]) ** 2) / float(Wnorm2))
    S = S.T
    P = P.T  # row: freq col: time to get conventional spectrogram orientation

    Th = float(Hop) / float(sample_rate)
    T = np.arange(0, (NumBlock) * Th, Th)

    Ls1, Ls2 = S.shape
    m1 = int(np.floor(zp1 / Hop))
    m2 = int(np.ceil((zp1 + zp2) / Hop))
    S = S[:, m1:Ls2 - m2]
    P = P[:, m1:Ls2 - m2]
    T = T[m1:Ls2 - m2]

    return S, P, freq, T




def plot_stft(signal, file_name, win_length=None, hop_length=None,
              window_type=None, sample_rate=None, n_fft_bins=None,
              freq_max=None, show_interactive_plot=False):
    """ Plots a stft of signal with the given window attributes

    Parameters:
        signal (np.array):
        file_name (str):
        num_ffts (Optional[int]): min number of desired freq. samples in (-pi,pi]. MUST be >= L. Defaults to
         int(2 ** np.ceil(np.log2(win_length)))
        freq_max (int): Max frequency to display
        window_attributes (Optional[WindowAttributes]): Contains all info about windowing for stft.
        win_length (Optional[int]): length of one window (in # of samples)
        window_type (Optional[WindowType]): window type
        win_overlap (Optional[int]): number of overlapping samples between adjacent windows
        sample_rate (int): sampling rate of the signal
        show_interactive_plot (Optional[bool]): Flag indicating if plot should be shown when function is run.
         Defaults to False

    Note:
         Either stft_params or all of [win_length, window_type, window_overlap, and num_ffts] must be provided.

    """
    sample_rate = Constants.DEFAULT_SAMPLE_RATE if sample_rate is None else sample_rate

    required = [win_length, hop_length, window_type]
    if any([r is None for r in required]):
        defaults = StftParams(sample_rate)

        win_length = defaults.window_length if win_length is None else win_length
        hop_length = defaults.hop_length if hop_length is None else hop_length
        window_type = defaults.window_type if window_type is None else window_type


    (stft, psd, freqs, time) = e_stft_plus(signal, win_length, hop_length, window_type, sample_rate, n_fft_bins)

    freq_max = Constants.MAX_FREQUENCY if freq_max is None else freq_max

    # TODO: this can be better!
    time_tile = np.tile(time, (len(freqs), 1))
    freq_tile = np.tile(freqs.T, (len(time), 1)).T
    sp = 10 * np.log10(np.abs(psd))
    sp = sp.T[:(len(sp.T)/2 + 1)]
    plt.pcolormesh(time_tile, freq_tile, sp)

    plt.axis('tight')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.xlim(time[0], time[-1])
    plt.ylim(freqs[0], freq_max)

    # plt.specgram(signal, NFFT=n_fft_bins, Fs=sample_rate) #, Fc=freq_max) # , window=window_type)
    plt.savefig(file_name)

    if show_interactive_plot:
        plt.interactive('True')
        plt.show()


def f_istft(stft, win_length=None, window_type=None, win_overlap=None, sample_rate=None, window_attributes=None):
    """Computes the inverse STFT of a spectrogram using Overlap Addition method

    Parameters:
        stft (np.array): one-sided stft (spectrogram) of the signal x
        win_length (Optional[int]): length of one window (in # of samples)
        window_type (Optional[WindowType]): window type
        win_overlap (Optional[int]): number of overlapping samples between adjacent windows
        sample_rate (int): sampling rate of the signal
        window_attributes (StftParams): WindowAttributes object that has all of the windowing info

    Returns:
        * **t** (*np.array*) - Numpy array containing time values for the reconstructed signal
        * **y** (*np.array*) - Numpy array containing the reconstructed signal
    """

    if window_attributes is None:
        if all(i is None for i in [win_length, window_type, win_overlap, sample_rate]):
            raise Exception(
                'Cannot do f_istft()! win_length, window_type, window_overlap, sample_rate are all required!')
    else:
        win_length = window_attributes.window_length
        window_type = window_attributes.window_type
        win_overlap = window_attributes.window_overlap_samples
        sample_rate = window_attributes.sample_rate


    # Get spectrogram dimensions and compute window hop size
    Nc = stft.shape[1]  # number of columns of X
    Hop = int(win_length - win_overlap)

    # Form the full spectrogram (-pi,pi]
    Xext = stft[-2:0:-1, :]
    X_inv_conj = Xext.conj()
    stft = np.vstack([stft, X_inv_conj])

    W = make_window(window_type, win_length)

    # Reconstruction through OLA
    Ly = int((Nc - 1) * Hop + win_length)
    y = np.zeros((1, Ly))

    for h in range(0, Nc):
        yh = np.real(spfft.ifft(stft[:, h]))
        hh = int(h * Hop)
        yh = yh.reshape(y[0, hh:hh + win_length].shape)
        y[0, hh:hh + win_length] = y[0, hh:hh + win_length] + yh[0:win_length]

    c = sum(W) / Hop
    y = y[0, :] / c
    t = np.arange(Ly) / float(sample_rate)

    return y, t

def e_stft(signal, window_length, hop_length, window_type, n_fft_bins=None):
    if n_fft_bins is None:
        n_fft_bins = window_length

    n_hops = int(np.ceil(float(len(signal)) / hop_length))

    # zero pad signal
    if n_hops * hop_length >= len(signal):
        sig_temp = np.zeros((n_hops + 1) * hop_length + window_length)
        sig_temp[0:len(signal)] += signal
        signal = sig_temp

    window = make_window(window_type, window_length)

    stft = np.zeros((n_hops, n_fft_bins), dtype=complex)
    for hop in range(n_hops):
        start = hop * hop_length
        end = start + window_length
        unwindowed_signal = signal[start:end]
        windowed_signal = np.multiply(unwindowed_signal, window)
        stft[hop, ] = spfft.fft(windowed_signal, n=n_fft_bins)

    return stft

def e_istft(stft, window_length, hop_length, window_type, n_fft_bins=None):
    if n_fft_bins is None:
        n_fft_bins = window_length

    n_hops = len(stft)
    window_length = len(stft[0])
    signal_length = (n_hops - 1) * hop_length + window_length
    signal = np.zeros(signal_length)

    for n in range(n_hops):
        start = n * hop_length
        end = start + window_length
        signal[start:end] = signal[start:end] + np.real(spfft.ifft(stft[n], n=n_fft_bins))

    return signal

def e_stft_plus(signal, window_length, hop_length, window_type, sample_rate, n_fft_bins=None):
    """
    Does a short time fourier transform (STFT) of the signal (by calling e_stft() ), but also calculates
    the power spectral density (PSD), frequency and time vectors for the calculated STFT.
    :param signal:
    :param window_length:
    :param hop_length:
    :param window_type:
    :param sample_rate:
    :param n_fft_bins:
    :return:
    """
    if n_fft_bins is None:
        n_fft_bins = window_length

    stft = e_stft(signal, window_length, hop_length, window_type, n_fft_bins)
    frequency_vector = (sample_rate / 2) * np.linspace(0, 1, (n_fft_bins / 2) + 1)

    time_vector = np.array(range(len(stft)))
    hop_in_secs = hop_length / (1.0 * sample_rate)
    time_vector = time_vector * hop_in_secs

    window = make_window(window_type, window_length)
    win_dot = np.dot(window, window.T)
    psd = np.zeros_like(stft, dtype=float)
    for i in range(len(psd)):
        psd[i, :] = (1 / float(sample_rate)) * ((abs(stft[i, :]) ** 2) / float(win_dot))

    return stft, psd, frequency_vector, time_vector

def make_window(window_type, length):
    """Returns an np array of type window_type

    Parameters:
        window_type (WindowType): Type of window to create, window_type object
        length (int): length of window
    Returns:
         window (np.array): np array of window_type
    """

    # Generate samples of a normalized window
    if window_type == WindowType.RECTANGULAR:
        return np.ones(length)
    elif window_type == WindowType.HANN:
        return hann(length, False)
    elif window_type == WindowType.BLACKMAN:
        return blackman(length, False)
    elif window_type == WindowType.HAMMING:
        return hamming(length, False)
    else:
        return None


class WindowType:
    RECTANGULAR = 'rectangular'
    HAMMING = 'hamming'
    HANN = 'hann'
    BLACKMAN = 'blackman'
    DEFAULT = HAMMING

    all_types = [RECTANGULAR, HAMMING, HANN, BLACKMAN]

    def __init__(self):
        pass


class StftParams(object):
    """
    The StftParams class is a container for information needed to run an STFT or iSTFT.
    This object will get passed around instead of each of these individual attributes.
    """

    def __init__(self, sample_rate, window_length=None, hop_length=None, window_type=None, n_fft_bins=None):
        self.sample_rate = sample_rate

        # default to 40ms windows
        default_win_len = int(2 ** (np.ceil(np.log2(Constants.DEFAULT_WIN_LEN_PARAM * sample_rate))))
        self._window_length = default_win_len if window_length is None else window_length
        self._hop_length = self._window_length / 2 if hop_length is None else hop_length
        self.window_type = WindowType.DEFAULT if window_type is None else window_type
        self._n_fft_bins = self._window_length if n_fft_bins is None else n_fft_bins

        self._hop_length_needs_update = True
        self._n_fft_bins_needs_update = True

        if hop_length is not None:
            self._hop_length_needs_update = False

        if n_fft_bins is not None:
            self._n_fft_bins_needs_update = False

    @property
    def window_length(self):
        return self._window_length

    @window_length.setter
    def window_length(self, value):
        """
        Length of window in samples. If window_overlap or num_fft are not set manually,
        then changing this will update them to hop_length = window_length / 2, and
        and num_fft = window_length
        :param value:
        :return:
        """
        self._window_length = value

        if self._n_fft_bins_needs_update:
            self._n_fft_bins = value

        if self._hop_length_needs_update:
            self._hop_length = value / 2

    @property
    def hop_length(self):
        return self._hop_length

    @hop_length.setter
    def hop_length(self, value):
        self._hop_length_needs_update = False
        self._hop_length = value

    @property
    def n_fft_bins(self):
        return self._n_fft_bins

    @n_fft_bins.setter
    def n_fft_bins(self, value):
        """
        Number of FFT bins per stft window.
        By default this is linked to window_length (value of window_length),
        but if this is set manually then they are both independent.
        :param value:
        :return:
        """
        self._n_fft_bins_needs_update = False
        self._n_fft_bins = value
