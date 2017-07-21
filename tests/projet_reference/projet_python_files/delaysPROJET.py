# -*- coding: utf-8 -*-
"""
Implements PROJET for separating spatial audio, including delays
as described in the paper:

@article{fitzgeraldPROJETb,
  TITLE = {{Projection-based demixing of spatial audio}},
  AUTHOR = {D. Fitzgerald and A. Liutkus and R. Badeau},
  JOURNAL = {{IEEE Transactions on Audio, Speech and Language Processing}},
  PUBLISHER = {{Institute of Electrical and Electronics Engineers}},
  YEAR = {2016},
  MONTH = May,
}

the main function is: separate

--------------------------------------------------------------------------
Copyright (c) 2016, Antoine Liutkus, Inria
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""



# importing stuff
import numpy as np
from basics import wav, stft
import os
from joblib import Parallel, delayed

# check for the support of fast BLAS
try:
    import numpy.core._dotblas
    print 'FAST BLAS'
except ImportError:
    print 'Slow BLAS'

# defining global variables for joblib parallelization
JOBLIB_NCORES = 5
JOBLIB_TEMPFOLDER = None
JOBLIB_BACKEND = 'threading' #None  #
JOBLIB_VERBOSE = 0

# define small epsilon value for regularization (divide by zero errors)
eps = 1e-20


def get_vf(Xf, Nf):
    """
    compute the 1-spectrogram of the projection of a frequency band of the mix at 1 frequency on some directions
    :param   Xf:  T x I        complex STFT of mix at a given f
    :param   Nf:  Mp x Md x I  projection matrix
    :return: Vf:  Mp x Ml x Nt magnitude spectrogram of projection
    """
    Vf = np.tensordot(Nf, Xf, axes=(-1, 1))
    Vf = np.abs(Vf)
    return Vf


def get_Kf(Nf, thetas_f):
    """
    builds the Kf tensor based on the projection matrix Nf and
    the angles
    :param Nf:       Mp x Md x I        projection matrix
    :param thetas_f: Lp x Ld x I        locations set
    :return Kf     : Mp x Md x Lp x Ld  tensor Kf, assuming alpha=1
    """
    Kf = np.tensordot(Nf, thetas_f, axes=((2, 2)))
    Kf = np.abs(Kf)
    return Kf


def get_anechoic(phis, taus, nfft):
    """builds the frequency response theta(f) for the different phis and taus

    :param  phis: panning angles in radian (array of lenght Np)
    :param  taus: delays in samples (array of length Nd)
    :param  nfft: number of fft points
    :return       F x Np x Nd x 2 frequency response for all combinations
    """
    F = (nfft + 1) / 2 if nfft % 2 else  (nfft / 2) + 1
    Np = len(phis)
    Nd = len(taus)

    thetas = np.zeros((F, Np, Nd, 2),dtype=complex)
    ones = np.ones((F,))
    for itau, tau in enumerate(taus):
        theta_temp = np.exp(-1j * 2 * np.pi * np.linspace(0, 0.5, F) * tau)
        thetas[:, :, itau, 0] = np.outer(ones, np.cos(phis))
        thetas[:, :, itau, 1] = np.outer(theta_temp, np.sin(phis))
    return thetas


def updates_f(Q, Xf, Nf, thetas_f, Pf, alpha=1, beta=1):
    """computing the constants for this frequency
    :param Q:       Lp x Ld x J  locations gains
    :param Xf:      T  x I       comlex STFT of mix at frequency
    :param Nf:      Mp x Md x I  projection matrix
    :param thetas_f:Lp x Ld x I  location set
    :param Pf:      T  x J       PSD estimates of sources
    :param alpha:               for alpha-harmonizable model
    :param beta:                divergence to use
    :return (Pf, [Qnum_f,Qdenum_f]) new Pf and the contribution of this frequency to the update of Q
    """

    # get Kf and Vf for this frequency
    Kf = get_Kf(Nf, thetas_f) ** alpha
    vf = eps + get_vf(Xf, Nf) ** alpha

    #1/ UPDATE Pf
    # compute current model for the projections
    KfQ = np.tensordot(Kf, Q, axes=2)  # Mp x Md x J
    sigma_f = eps + np.tensordot(KfQ, Pf, axes=([2, 1]))  # Mp x Md x T

    #compute the Mp x Md x T tensor to use at numerator, and denominator
    if beta==1:
        temps = (sigma_f ** (beta - 2) * vf, np.ones(sigma_f.shape,dtype="float32"))
    else:
        temps = (sigma_f ** (beta - 2) * vf, sigma_f ** (beta - 1))

    #compute numerator and denominator
    Pf_num_denum = [eps + np.tensordot(KfQ, M, axes=([0, 1], [0, 1])) for M in temps]

    #perform multiplicative update
    Pf *= (Pf_num_denum[0] / Pf_num_denum[1]).T


    #1/ UPDATE Q
    # compute current model for the projections
    sigma_f = eps + np.tensordot(KfQ, Pf, axes=(2, 1))  # MpxMdxNt

    #compute the Mp x Md x T tensor to use at numerator, and denominator
    if beta==1:
        temps = [sigma_f ** (beta - 2) * vf, np.ones(sigma_f.shape,dtype="float32")]
    else:
        temps = [sigma_f ** (beta - 2) * vf, sigma_f ** (beta - 1)]
    #compute numerator and denominator for this frequency
    Q_num_denum_f = [np.tensordot(Kf,  # Mp x Md x Np x Nd
                                  np.tensordot(M, Pf, axes=(2, 0)),  # Mp x Md x J
                                  axes=([0, 1], [0, 1]))
                     for M in temps]
    return Pf, Q_num_denum_f


def separate_f(Xf, Pf, Q, Nf, thetas_f, alpha):
    """separate the projections and recover image estimates for this frequency

    :param Xf:      T  x I       comlex STFT of mix at frequency    :param Pf:
    :param Pf:      T  x J       PSD estimates of sources
    :param Q:       Lp x Ld x J  locations gains
    :param Nf:      Mp x Md x I  projection matrix
    :param thetas_f:Lp x Ld x I  location set
    :param alpha:               for alpha-harmonizable model
    :return yj_f:   T x I x J   images for this frequency
    """
    # get the Kf
    Kf = get_Kf(Nf, thetas_f) ** alpha

    #reshape everything as matrices
    (T, I) = Xf.shape

    (Mp, Md) = Nf.shape[:2]
    (Lp, Ld, J) = Q.shape

    M = Mp * Md
    L = Lp * Ld

    Nf = np.reshape(Nf, (M, I))
    Kf = np.reshape(Kf, (M, L))
    Q = np.reshape(Q, (L, J))

    #compute Kf Q
    KfQ = np.dot(Kf, Q)  # M x J

    #compute the wiener gains to apply on projections
    wiener_j_f = KfQ[:, None, :] * Pf[None, ...]  # M x T x J
    wiener_j_f /= (eps + np.sum(wiener_j_f, axis=-1)[..., None])

    #compute projected mixture and separate the proj into cj
    cj_f = np.dot(Nf, Xf.T)[..., None] * wiener_j_f  # M x T x J

    #recompute the sources images by inverting the projection
    Nf_inv = np.linalg.pinv(Nf)  # IxM
    return np.swapaxes(np.tensordot(Nf_inv, cj_f, axes=(1, 0)),  # I x T x J
                       0, 1)  # T x I x J


def separate(inputFilename, outputDir, J, Lp, Ld, Mp, Md, alpha=1, beta=1, nIter=200, maxDelaySamples=5, firstCentered=True):
    """ performs separation using PROJET algorithm, with delays
    :param inputFilename:   path to wavfile to separate
    :param outputDir:       directory to which output the separated files
    :param J:               number of sources
    :param Lp:              number of panning values (from 0 to 2pi)
    :param Ld:              number of delays to use
    :param Mp:              number of panning values for projections
    :param Md:              number of delays for projections
    :param alpha:           for the alpha-harmonizable model
    :param beta:            beta divergence to use
    :param nIter:           number of iterations of PROJET
    :param maxDelaySamples: the delays will span [-maxDelaySamples +mayDelaySamples]
    """

    #STFT parameters
    nfft = 1024
    overlap = 0.75#0.75
    maxLength = 20


    # 1) Loading data
    # ----------------
    basenameFile = os.path.split(inputFilename)[1]
    (sig, fs) = wav.wavread(inputFilename, maxLength)
    fs = float(fs)

    print 'Launching STFT'
    hop = float(nfft) * (1.0 - overlap)
    X = stft.stft(sig, nfft, hop, real=True, verbose=False)
    (Nf, Nt, I) = X.shape

    # initialize parameters to random
    P = np.abs(np.random.randn(Nf, Nt, J))
    Q = np.abs(np.random.randn(Lp, Ld, J))
    if firstCentered:
        Q[Lp/2,:,0]=4.0

    # compute panning and projection tensors
    if Mp>1:
        pannings=np.linspace(0, -np.pi / 2.0, Mp)
    else:
        pannings = [np.pi/4.0]
    N = get_anechoic(pannings, np.linspace(-maxDelaySamples, maxDelaySamples, Md), nfft)
    if Lp>1:
        pannings=np.linspace(0, np.pi / 2.0, Lp)
    else:
        pannings = [np.pi/4.0]
    thetas = get_anechoic(pannings, np.linspace(-maxDelaySamples, maxDelaySamples, Ld), nfft)

    # iterations
    for iteration in range(nIter):
        print 'PROJET [%d/%d]' % (iteration + 1, nIter)
        #Launching parallel updates for each frequency
        P, Q_num_denum = zip(*(Parallel(n_jobs=JOBLIB_NCORES,
                                        verbose=JOBLIB_VERBOSE,
                                        backend=JOBLIB_BACKEND,
                                        temp_folder=JOBLIB_TEMPFOLDER)(
            delayed(updates_f)(Q, X[f, ...], N[f, ...], thetas[f, ...], P[f, ...], alpha, beta) for f in range(Nf))))

        #gather P
        P = np.array(P)

        #sum all contributions for Q and apply them
        Q_num_denum = np.array(Q_num_denum)
        Q_num_denum = np.sum(np.array(Q_num_denum), axis=0)
        Q *= Q_num_denum[0, ...] / Q_num_denum[1, ...]

        if (not (iteration % 30) and (iteration >= 50)) or (iteration == nIter - 1):
            # separating from time to time
            print 'separation...'
            Y = np.array(Parallel(n_jobs=JOBLIB_NCORES,
                                  verbose=JOBLIB_VERBOSE,
                                  backend=JOBLIB_BACKEND,
                                  temp_folder=JOBLIB_TEMPFOLDER)(
                delayed(separate_f)(X[f, ...], P[f, ...], Q, N[f, ...], thetas[f, ...], alpha) for f in range(Nf)))

            for j in range(J):
                print '    source %d/%d' % (j + 1, J)
                yj = stft.istft(Y[..., j], 1, hop, real=True, shape=sig.shape).astype(np.float32)
                sourceFilename = os.path.join(outputDir, '%s_source_%d' % (basenameFile, j + 1) + '.wav')
                wav.wavwrite(yj, fs, sourceFilename, verbose=False)


