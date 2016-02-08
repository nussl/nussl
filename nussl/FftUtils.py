#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from WindowType import WindowType
import scipy.fftpack as spfft


def f_stft(signal, num_ffts=None, window_attributes=None, win_length=None, window_type=None, window_overlap=None,
           sample_rate=None):
    """Computes the one-sided STFT of a signal

    Parameters:
        signal (np.array): row vector containing the signal.
        num_ffts (Optional[int]): min number of desired freq. samples in (-pi,pi]. MUST be >= L. Defaults to
         int(2 ** np.ceil(np.log2(win_length)))
        window_attributes (Optional[WindowAttributes]): Contains all info about windowing for stft.
        win_length (Optional[int]): length of one window (in # of samples)
        window_type (Optional[WindowType]): window type
        window_overlap (Optional[int]): number of overlapping samples between adjacent windows
        sample_rate (int): sampling rate of the signal

    Note:
        Either window_attributes or all of [win_length, window_type, window_overlap, and num_ffts] must be provided.

    Returns:
        * **S** (*np.array*) - 2D numpy matrix containing the one-sided short-time Fourier transform of the signal
         (complex)
        * **P** (*np.array*) - 2D numpy matrix containing the one-sided PSD of the signal
        * **F** (*np.array*) - frequency vector
        * **T** (*np.array*) - time vector
    """

    if window_attributes is None:
        if all(i is None for i in [win_length, window_type, window_overlap, num_ffts, sample_rate]):
            raise Exception(
                'Cannot do f_stft()! win_length, window_type, window_overlap, num_ffts, sample_rate are all required!')
    else:
        win_length = window_attributes.window_length
        window_type = window_attributes.window_type
        window_overlap = window_attributes.window_overlap_samples
        num_ffts = int(2 ** np.ceil(np.log2(win_length)))
        sample_rate = window_attributes.sample_rate

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


def plot_stft(signal, file_name, num_ffts=None, freq_max=None, window_attributes=None, win_length=None,
              window_type=None, win_overlap=None, sample_rate=None, show_interactive_plot=False):
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
         Either window_attributes or all of [win_length, window_type, window_overlap, and num_ffts] must be provided.

    """
    if window_attributes is None:
        if all(i is None for i in [win_length, window_type, win_overlap, num_ffts]):
            raise Exception(
                'Cannot do plot_stft()! win_length, window_type, window_overlap, num_ffts are all required!')

        (S, P, F, Time) = f_stft(signal, num_ffts=num_ffts, window_attributes=window_attributes, win_length=win_length,
                                 window_type=window_type, window_overlap=win_overlap, sample_rate=sample_rate)
    else:
        (S, P, F, Time) = f_stft(signal, window_attributes=window_attributes)

    TT = np.tile(Time, (len(F), 1))
    FF = np.tile(F.T, (len(Time), 1)).T
    SP = 10 * np.log10(np.abs(P))
    plt.pcolormesh(TT, FF, SP)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.xlim(Time[0], Time[-1])
    plt.ylim(F[0], freq_max)
    plt.savefig(file_name)

    if show_interactive_plot:
        plt.interactive('True')
        plt.show()


def f_istft(stft, win_length=None, window_type=None, win_overlap=None, sample_rate=None, window_attributes=None):
    """Computes the inverse STFT of a spectrogram using Overlap Addition method

    Parameters:
        stft (np.array): one-sided do_STFT (spectrogram) of the signal x
        win_length (Optional[int]): length of one window (in # of samples)
        window_type (Optional[WindowType]): window type
        win_overlap (Optional[int]): number of overlapping samples between adjacent windows
        sample_rate (int): sampling rate of the signal
        window_attributes (WindowAttributes): WindowAttributes object that has all of the windowing info

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


def make_window(window_type, length):
    """Returns an np array of type window_type

    Parameters:
        window_type (WindowType): Type of window to create, window_type object
        length (int): length of window
    Returns:
         window (np.array): np array of windowtype
    """

    # Generate samples of a normalized window
    if (window_type == WindowType.RECTANGULAR):
        return np.ones(length)
    elif (window_type == WindowType.HANN):
        return np.hanning(length)
    elif (window_type == WindowType.BLACKMAN):
        return np.blackman(length)
    else:
        return np.hamming(length)
