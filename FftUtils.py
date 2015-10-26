import numpy as np
import matplotlib.pyplot as plt
from WindowType import WindowType

plt.interactive('True')
import scipy.fftpack as spfft


def f_stft(signal, winLength, windowType, winOverlap, sampleRate, nFfts=None, mkplot=0, fmax=None):
    """
    This function computes the one-sided STFT of a signal
    :param signal: signal, row vector
    :param winLength: length of one window (in # of samples)
    :param windowType: window type, WindowType object
    :param winOverlap: number of overlapping samples between adjacent windows
    :param sampleRate: sampling rate of the signal
    :param nFfts: min number of desired freq. samples in (-pi,pi]. MUST be >= L.
    :param mkplot: binary input (1 for show plot). Default value is 0
    :param fmax: maximum frequency shown on the spectrogram if mkplot is 1. If not specified
                it will be set to fs/2.
    :return:    S: 2D numpy matrix containing the one-sided short-time Fourier transform of the signal (complex)
                P: 2D numpy matrix containing the one-sided PSD of the signal
                F: frequency vector
                T: time vector
    """
    """
    *NOTE* The default value for nfft is the next power 2 of the window length (nextpower2(L)).
       e.g. if nfft is not specified and L=257, nfft will be set to 512.
    mkplot:
    fmax(optional):

    Outputs:

    * Note: windowing and fft will be performed row-wise so that the code runs faster

    """
    if nFfts is None:
        nFfts = int(2 ** np.ceil(np.log2(winLength)))
    if mkplot == 1 and fmax is None:
        fmax = sampleRate / 2

    signal = np.mat(signal)

    # split data into blocks (make sure X is a row vector)
    if signal.shape[0] != 1:  # TODO: X.ndim? need a better check here
        raise ValueError('X must be a row vector')
    elif nFfts < winLength:
        raise ValueError('nfft must be greater or equal the window length (L)!')

    Hop = int(winLength - winOverlap)
    N = signal.shape[1]

    # zero-pad the vector at the beginning and end to reduce the window tapering effect
    if np.mod(winLength, 2) == 0:
        zp1 = winLength / 2
    else:
        zp1 = (winLength - 1) / 2

    signal = np.hstack([np.zeros((1, zp1)), signal, np.zeros((1, zp1))])
    N = N + 2 * zp1

    # zero pad if N-2*zp1 is not an integer multiple of Hop
    rr = np.mod(N - 2 * zp1, Hop)
    if rr != 0:
        zp2 = Hop - rr
        signal = np.hstack([signal, np.zeros((1, zp2))])
        N = signal.shape[1]
    else:
        zp2 = 0

    NumBlock = int(((N - winLength) / Hop) + 1)

    W = MakeWindow(windowType, winLength)
    Wnorm2 = np.dot(W, W.T)

    # Generate freq. vector
    F = (sampleRate / 2) * np.linspace(0, 1, num=nFfts / 2 + 1)
    Lf = len(F)

    # Take the fft of each block
    S = 1j * np.zeros((NumBlock, Lf))  # row: time, col: freq. to increase speed
    P = np.zeros((NumBlock, Lf))

    for i in range(0, NumBlock):
        Xw = np.multiply(W, signal[0, (i * Hop):(i * Hop + winLength)])
        XX = spfft.fft(Xw, n=nFfts)
        XX_trun = XX[0, 0:Lf]

        S[i, :] = XX_trun
        P[i, :] = (1 / float(sampleRate)) * ((abs(S[i, :]) ** 2) / float(Wnorm2))
    S = S.T
    P = P.T  # row: freq col: time to get conventional spectrogram orientation

    Th = float(Hop) / float(sampleRate)
    T = np.arange(0, (NumBlock) * Th, Th)

    Ls1, Ls2 = S.shape
    m1 = int(np.floor(zp1 / Hop))
    m2 = int(np.ceil((zp1 + zp2) / Hop))
    S = S[:, m1:Ls2 - m2]
    P = P[:, m1:Ls2 - m2]
    T = T[m1:Ls2 - m2]

    # plot if specified
    if mkplot == 1:
        TT = np.tile(T, (len(F), 1))
        FF = np.tile(F.T, (len(T), 1)).T
        SP = 10 * np.log10(np.abs(P))
        plt.pcolormesh(TT, FF, SP)
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.xlim(T[0], T[-1])
        plt.ylim(F[0], fmax)
        plt.show()

    return S, P, F, T


def f_istft(stft, winLength, windowType, winOverlap, sampleRate):
    """
    This function computes the inverse STFT of a spectrogram using
    Overlap Addition method

    :param stft: one-sided STFT (spectrogram) of the signal x
    :param winLength: window length (in # of samples)
    :param windowType: window type, (string): 'Rectangular', 'Hamming', 'Hanning', 'Blackman'
    :param winOverlap: overlap between adjacent windows in STFT analysis
    :param sampleRate: sampling rate of the original signal x
    :return:    t: Numpy array containing time values for the reconstructed signal
                y: Numpy array containing the reconstructed signal
    """
    # Get spectrogram dimenstions and compute window hop size
    Nc = stft.shape[1]  # number of columns of X
    Hop = int(winLength - winOverlap)

    # Form the full spectrogram (-pi,pi]
    Xext = stft[-2:0:-1, :]
    X_inv_conj = Xext.conj()
    stft = np.vstack([stft, X_inv_conj])

    W = MakeWindow(windowType, winLength)

    ## Reconstruction through OLA
    Ly = int((Nc - 1) * Hop + winLength)
    y = np.zeros((1, Ly))

    for h in range(0, Nc):
        yh = np.real(spfft.ifft(stft[:, h]))
        hh = int(h * Hop)
        y[0, hh:hh + winLength] = y[0, hh:hh + winLength] + yh[0:winLength]

    c = sum(W) / Hop
    y = y[0, :] / c
    t = np.arange(Ly) / float(sampleRate)

    return y, t


def MakeWindow(windowType, L):
    """
    Returns an np array of type windowType
    :param windowType: Type of window to create, WindowType object
    :param L: length of window
    :return: np array of window
    """

    # Generate samples of a normalized window
    if (windowType == WindowType.RECTANGULAR):
        return np.ones(L)
    elif (windowType == WindowType.HANNING):
        return np.hanning(L)
    elif (windowType == WindowType.BLACKMAN):
        return np.blackman(L)
    else:
        return np.hamming(L)
