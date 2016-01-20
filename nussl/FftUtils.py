#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from WindowType import WindowType

plt.interactive('True')
import scipy.fftpack as spfft


def f_stft(signal, num_ffts=None, window_attributes=None, win_length=None, window_type=None, window_overlap=None,
           sample_rate=None):
    """
    This function computes the one-sided do_STFT of a signal
    :param window_attributes: window_attributes object that contains all info about windowing
    :param signal: signal, row vector
    :param win_length: length of one window (in # of samples)
    :param window_type: window type, window_type object
    :param window_overlap: number of overlapping samples between adjacent windows
    :param sample_rate: sampling rate of the signal
    :param num_ffts: min number of desired freq. samples in (-pi,pi]. MUST be >= L.
    :param mkplot: binary input (1 for show plot). Default value is 0
    :return:    S: 2D numpy matrix containing the one-sided short-time Fourier transform of the signal (complex)
                P: 2D numpy matrix containing the one-sided PSD of the signal
                F: frequency vector
                T: time vector
    """
    """
    *NOTE* The default value for num_fft_bins is the next power 2 of the window length (nextpower2(L)).
       e.g. if num_fft_bins is not specified and L=257, num_fft_bins will be set to 512.
    mkplot:
    freq_max(optional):

    Outputs:

    * Note: windowing and fft will be performed row-wise so that the code runs faster

    """
    if window_attributes is None:
        if all(i is None for i in [win_length, window_type, window_overlap, num_ffts]):
            raise Exception('Cannot do do_STFT! win_length, window_type, window_overlap, num_ffts are all required!')
    else:
        win_length = window_attributes.window_length
        window_type = window_attributes.window_type
        window_overlap = window_attributes.window_overlap
        num_ffts = int(2 ** np.ceil(np.log2(win_length)))

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

    # TODO: don't window


def PlotStft(signal, file_name, num_ffts=None, freq_max=None, window_attributes=None, win_length=None, window_type=None,
             win_overlap=None, sample_rate=None, show_interactive_plot=False):
    (S, P, F, Time) = f_stft(signal, num_ffts=num_ffts, window_attributes=window_attributes, win_length=win_length,
                             window_type=window_type, window_overlap=win_overlap, sample_rate=sample_rate)
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
        plt.show()


def f_istft(stft, win_length, window_type, win_overlap, sample_rate):
    """
    This function computes the inverse do_STFT of a spectrogram using
    Overlap Addition method

    :param stft: one-sided do_STFT (spectrogram) of the signal x
    :param win_length: window length (in # of samples)
    :param window_type: window type, (string): 'Rectangular', 'Hamming', 'Hanning', 'Blackman'
    :param win_overlap: overlap between adjacent windows in do_STFT analysis
    :param sample_rate: sampling rate of the original signal x
    :return:    t: Numpy array containing time values for the reconstructed signal
                y: Numpy array containing the reconstructed signal
    """
    # TODO: make this accept a WindowAttributes object
    # Get spectrogram dimensions and compute window hop size
    Nc = stft.shape[1]  # number of columns of X
    Hop = int(win_length - win_overlap)

    # Form the full spectrogram (-pi,pi]
    Xext = stft[-2:0:-1, :]
    X_inv_conj = Xext.conj()
    stft = np.vstack([stft, X_inv_conj])

    W = make_window(window_type, win_length)

    ## Reconstruction through OLA
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
    """
    Returns an np array of type window_type
    :param window_type: Type of window to create, window_type object
    :param length: length of window
    :return: np array of window
    """

    # Generate samples of a normalized window
    if (window_type == WindowType.RECTANGULAR):
        return np.ones(length)
    elif (window_type == WindowType.HANNING):
        return np.hanning(length)
    elif (window_type == WindowType.BLACKMAN):
        return np.blackman(length)
    else:
        return np.hamming(length)
