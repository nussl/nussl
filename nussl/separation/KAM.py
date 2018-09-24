#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

 *** NOTE: KAM has not yet been updated to work with the current nussl framework yet. ***

This module implements the Kernel Additive Modeling (KAM) algorithm and its light
version (KAML) for source separation. 

References:
[1] Liutkus, Antoine, et al. "Kernel additive models for source separation." 
    Signal Processing, IEEE Transactions on 62.16 (2014): 4298-4310.
[2] Liutkus, Antoine, Derry Fitzgerald, and Zafar Rafii. "Scalable audio 
    separation with light kernel additive modelling." IEEE International 
    Conference on Acoustics, Speech and Signal Processing (ICASSP). 2015.

"""

import numpy as np
import matplotlib.pyplot as plt

plt.interactive('True')
import scipy.ndimage.filters
import scipy
from ..core.audio_signal import AudioSignal


def kam(Inputfile, SourceKernels, Numit=1, SpecParams=np.array([]), FullKernel=False):
    """
    The 'kam' function implements the kernel backfitting algorithm to extract 
    J audio sources from I channel mixtures.
    
    Inputs: 
    Inputfile: (list) It can contain either:                 
               - Up to 3 elements: A string indicating the path of the .wav file containing 
                 the I-channel audio mixture as the first element. The second (optional) 
                 element indicates the length of the portion of the signal to be extracted 
                 in seconds(defult is the full lengths of the siganl) The third (optional) 
                 element indicates the starting point of the portion of the signal to be 
                 extracted (default is 0 sec).
               OR
               - 2 elements: An I-column Numpy matrix containing samples of the time-domain 
                 mixture as the first element and the sampling rate as the second element.
          
    SourceKernels: a list containg J sub-lists, each of which contains properties of 
                   one of source kernels. Kernel properties are:
                   -kernel type: (string) determines whether the kernel is one of the 
                                 pre-defined kernel types or a user-defined lambda function. 
                                 Choices are: 'cross','horizontal','vertical','periodic','userdef'
                   -kparams (for pre-defined kernels): a Numpy matrix containing the numerical 
                             values of the kernel parameters.
                   -knhood (for user-defined kernels): logical lambda function which defines 
                            receives the coordinates of two time-frequency bins and determines
                            whether they are neighbours (outputs TRUE if neighbour).
                   -kwfunc (optional): lambda function which receives the coordinates of two 
                            neighbouring time-frequency bins and computes the weight value at
                            the second bin given its distance from the first bin. The weight
                            values fall in the interval [0,1]. Default: all ones over the 
                            neighbourhood (binary kernel).
    
    Numit: (optional) number of iterations of the backfitting algorithm - default: 1     
                   
    SpecParams: (optional) structured containing spectrogram parameters including:
                         - windowtype (default is Hamming)
                         - windowlength (default is 60 ms)
                         - overlap_samples in [0,windowlength] (default is widowlength/2)
                         - num_fft_bins (default is windowlength)
                         - makeplot in {0,1} (default is 0)
                         - fmaxplot in Hz (default is fs/2)
                         example: 
                         SpecParams=np.zeros(1,dtype=[('windowtype','|S1'),
                                                      ('windowlength',int),
                                                      ('overlap_samples',int),
                                                      ('num_fft_bins',int),
                                                      ('makeplot',int),
                                                      ('fmaxplot',float)])
                         SpecParams['windowlength']=1024   

    FullKernel: (optional) binary input which determines the method used for median filtering.
               If the kernel has a limited support and the same shape over all time-freq. bins,
               then instead of the full kernel method a sliding window can be used in the median 
               filtering stage in order to make the algorithm run faster (linear computational
               complexity). The default value is False.
               A True value means implementing the case where the similarity measure is 
               computed for all possible combinations of TF bins, resulting in quadratic 
               computational complexity, and hence longer running time. 
                                        
        
    Outputs:
    shat: a Ls by I by J Numpy array containing J time-domain source images on I channels   
    fhat: a LF by LT by J Numpy array containing J power spectral dencities      
    """

    # Step (1): 
    # Load the audio mixture from the input path
    if len(Inputfile) == 2:
        Mixture = AudioSignal(audiosig=Inputfile[0], fs=Inputfile[1])
    elif len(Inputfile) == 3:
        Mixture = AudioSignal(file_name=Inputfile[0], siglen=Inputfile[1], sigstart=Inputfile[2])

    if len(SpecParams) != 0:
        for i in range(0, len(SpecParams.dtype)):
            nameTemp = SpecParams.dtype.names[i]
            valTemp = SpecParams[0][i]
            valTemp = str(valTemp)
            exec ('Mixture.' + nameTemp + '=' + valTemp)

    x, tvec = np.array([Mixture.x, Mixture.time])  # time-domain channel mixtures
    X, Px, Fvec, Tvec = Mixture.do_STFT()  # stft and PSD of the channel mixtures

    I = Mixture.numCh  # number of channel mixtures
    J = len(SourceKernels)  # number of sources
    LF = np.size(Fvec)  # length of the frequency vector
    LT = np.size(Tvec)  # length of the timeframe vector

    F_ind = np.arange(LF)  # frequency bin indices
    T_ind = np.arange(LT)  # time frame indices
    Tmesh, Fmesh = np.meshgrid(T_ind, F_ind)  # grid of time-freq indices for the median filtering step
    TFcoords = np.mat(np.zeros((LF * LT, 2), dtype=int))  # all time-freq index combinations
    TFcoords[:, 0] = np.mat(np.asarray(Fmesh.T).reshape(-1)).T
    TFcoords[:, 1] = np.mat(np.asarray(Tmesh.T).reshape(-1)).T


    # Generate source kernels:
    Kj = []
    for ns in range(0, J):
        SKj = SourceKernels[ns]
        if len(SKj) < 2:
            raise Exception('The information required for generating source kernels is insufficient.'
                            ' Each sub-list in SourceKernels must contain at least two elements.')
        KTYPE = SKj[0]
        if KTYPE != 'userdef' and len(SKj) == 2:
            Kj.append(Kernel(Type=SKj[0], ParamVal=SKj[1]))
        elif KTYPE != 'userdef' and len(SKj) == 3:
            Kj.append(Kernel(Type=SKj[0], ParamVal=SKj[1]), Wfunc=SKj[2])
        elif KTYPE == 'userdef' and len(SKj) == 2:
            Kj.append(Kernel(Type=SKj[0], Nhood=SKj[1]))
        elif KTYPE == 'userdef' and len(SKj) == 3:
            Kj.append(Kernel(Type=SKj[0], Nhood=SKj[1]), Wfunc=SKj[2])

    # Step (2): initialization
    # Initialize the PSDs with average mixture PSD and the spatial covarince matricies
    # with identity matrices

    X = np.reshape(X.T, (LF * LT, I))  # reshape the stft tensor into I vectors
    if I > 1:
        MeanPSD = np.mean(Px, axis=2) / (I * J)
    else:
        MeanPSD = Px / (J)
    MeanPSD = np.reshape(MeanPSD.T, (LF * LT, 1))  # reshape the mean PSD matrix into a vector

    fj = np.zeros((LF * LT, I * I, J))
    for j in range(0, J):
        fj[:, :, j] = np.tile(MeanPSD, (1, I * I))  # initialize by mean PSD

    Rj = 1j * np.zeros((1, I * I, J))
    for j in range(0, J):
        Rj[0, :, j] = np.reshape(np.eye(I), (1, I * I))
    Rj = np.tile(Rj, (LF * LT, 1, 1))

    ### Kernel Backfitting ###

    # start_time = time.clock()

    S = 1j * np.zeros((LF * LT, I, J))
    for n in range(0, Numit):

        # Step (3):
        # compute the inverse_mask term: [sum_j' f_j' R_j']^-1
        SumFR = np.sum(fj * Rj, axis=2)  ###  !!!!!!!!!! careful about memory storage!
        SumFR.shape = (LF * LT, I, I)
        SumFR += 1e-16 * np.random.randn(LF * LT, I, I)  # to avoid singularity issues

        InvSumFR = np.zeros((LF * LT, I, I), dtype='single')
        if I == 1:
            InvSumFR = 1 / SumFR
        elif I == 2:
            InvDet = 1 / (SumFR[:, 0, 0] * SumFR[1, 1] - SumFR[0, 1] * SumFR[1, 0])
            InvSumFR[:, 0, 0] = InvDet * SumFR[:, 1, 1]
            InvSumFR[:, 0, 1] = -InvDet * SumFR[:, 0, 1]
            InvSumFR[:, 1, 0] = -InvDet * SumFR[:, 1, 0]
            InvSumFR[:, 1, 1] = InvDet * SumFR[:, 0, 0]
        else:
            InvSumFR = np.linalg.inv(SumFR)
        InvSumFR.shape = (LF * LT, I * I)

        # compute sources, update PSDs and covariance matrices 
        for ns in range(0, J):
            FRinvsum = fj[:, :, ns] * Rj[:, :, ns] * InvSumFR
            Stemp = 1j * np.zeros((LF * LT, I))
            for nch in range(0, I):
                FRtemp = FRinvsum[:, nch * I:nch * I + 2]
                Stemp[:, nch] = np.sum(FRtemp * X, axis=1)
            S[:, :, ns] = Stemp

            # Step (4-a):
            Cj = np.repeat(Stemp, I, axis=1) * np.tile(np.conj(Stemp), (1, I))

            # Step (4-b):
            Cj_reshape = np.reshape(Cj, (LF * LT, I, I))
            Cj_trace = np.mat(np.matrix.trace(Cj_reshape.T)).T
            MeanCj = Cj / np.tile(Cj_trace, (1, I * I))
            MeanCj_reshape = np.reshape(np.array(MeanCj), (LF, LT, I * I), order='F')
            Rj[:, :, ns] = np.tile(np.sum(MeanCj_reshape, axis=1), (LT, 1))

            # Step (4-c):
            # Note: the summation over 't' at step 4-c in the 2014 paper is a typo!
            #       the correct formulation of zj is:
            #       zj=(1/I)*tr(inv(Rj(w)Cj(w,t)
            Rj_reshape = np.reshape(Rj[:, :, ns], (LF * LT, I, I))
            Rj_reshape += 1e-16 * np.random.randn(LF * LT, I, I)

            InvRj = np.zeros((LF * LT, I, I), dtype='single')
            if I == 1:
                InvRj = 1 / Rj_reshape
            elif I == 2:
                InvDetR = 1 / (Rj_reshape[:, 0, 0] * Rj_reshape[1, 1] - Rj_reshape[0, 1] * Rj_reshape[1, 0])
                InvRj[:, 0, 0] = InvDetR * Rj_reshape[:, 1, 1]
                InvRj[:, 0, 1] = -InvDetR * Rj_reshape[:, 0, 1]
                InvRj[:, 1, 0] = -InvDetR * Rj_reshape[:, 1, 0]
                InvRj[:, 1, 1] = InvDetR * Rj_reshape[:, 0, 0]
            else:
                InvRj = np.linalg.inv(Rj_reshape)
            InvRj.shape = (LF * LT, I * I)

            InvRjCj = np.reshape(InvRj * Cj, (LF * LT, I, I))
            zj = np.real(np.matrix.trace(InvRjCj.T) / I)
            zj = np.mat(zj)

            # Step (4-d):
            # Median filter the estimated PSDs

            # start_time = time.clock()
            Ktemp = Kj[ns]  # Kernel corresponding to the source #j

            if not FullKernel:  # kernel defined as a sliding window (faster method)
                centerF = (LF - np.mod(LF, 2)) / 2  # middle freq. bin
                centerT = (LT - np.mod(LT, 2)) / 2  # middle time frame
                centerTF = np.mat([centerF, centerT])  # middle time-freq. bin
                KWin = Ktemp.sim(centerTF, TFcoords)  # sliding kernel window
                KWin_reshape = np.reshape(KWin, (LT, LF)).T
                NZ = np.nonzero(KWin_reshape)  # range of element numbers of central nonzero elements
                KWin_shrink = KWin_reshape[NZ[0].min():NZ[0].max() + 1,
                              NZ[1].min():NZ[1].max() + 1]  # extract the central nonzero part
                ZMed = scipy.ndimage.filters.median_filter(np.reshape(zj, (LT, LF)).T,
                                                           footprint=KWin_shrink)  # median filter
                fj[:, :, ns] = np.reshape(ZMed.T, (LF * LT, 1))

            else:  # full kernel method (more general but slower approach)
                for ft in range(0, LF * LT):
                    simTemp = Ktemp.sim(TFcoords[ft, :], TFcoords)
                    NhoodTemp = np.nonzero(simTemp)
                    zjNhood = np.multiply(zj[NhoodTemp], simTemp[NhoodTemp])
                    fj[ft, :, ns] = np.median(np.array(zjNhood))

                    # print time.clock() - start_time, "seconds"


    # print time.clock() - start_time, "seconds"

    # Reshape the PSDs
    fhat = np.zeros((LF, LT, J))
    for ns in range(0, J):
        fhat[:, :, ns] = np.reshape(fj[:, 0, ns], (LT, LF)).T

        # Reshape the spectrograms
    Shat = 1j * np.zeros((LF, LT, I, J))  # estimated source STFTs
    for ns in range(0, J):
        for nch in range(0, I):
            Shat[:, :, nch, ns] = np.reshape(S[:, nch, ns], (LT, LF)).T


    # Compute the inverse_mask stft of the estimated sources
    shat = np.zeros((x.shape[0], I, J))
    sigTemp = AudioSignal()
    sigTemp.windowtype = Mixture.windowtype
    sigTemp.windowlength = Mixture.windowlength
    sigTemp.overlap_samples = Mixture.overlap_samples
    sigTemp.numCh = I
    for ns in range(0, J):
        sigTemp.X = Shat[:, :, :, ns]
        shat[:, :, ns] = sigTemp.istft()[0][0:x.shape[0]]

    return shat, fhat


def kaml(Inputfile, SourceKernels, AlgParams=np.array([10, 1]), Numit=1, SpecParams=np.array([]), FullKernel=False):
    """
    The 'kaml' function implements a computationally less expensive version of the 
    kernel backfitting algorithm. The KBF algorithm extracts J audio sources from 
    I channel mixtures.
    
    Inputs: 
    Inputfile: (list) It can contain either:                 
               - Up to 3 elements: A string indicating the path of the .wav file containing 
                 the I-channel audio mixture as the first element. The second (optional) 
                 element indicates the length of the portion of the signal to be extracted 
                 in seconds(defult is the full lengths of the siganl) The third (optional) 
                 element indicates the starting point of the portion of the signal to be 
                 extracted (default is 0 sec).
               OR
               - 2 elements: An I-column Numpy matrix containing samples of the time-domain 
                 mixture as the first element and the sampling rate as the second element.
          
    SourceKernels: a list containg J sub-lists, each of which contains properties of 
                   one of source kernels. Kernel properties are:
                   -kernel type: (string) determines whether the kernel is one of the 
                                 pre-defined kernel types or a user-defined lambda function. 
                                 Choices are: 'cross','horizontal','vertical','periodic','userdef'
                   -kparams (for pre-defined kernels): a Numpy matrix containing the numerical 
                             values of the kernel parameters.
                   -knhood (for user-defined kernels): logical lambda function which defines 
                            receives the coordinates of two time-frequency bins and determines
                            whether they are neighbours (outputs TRUE if neighbour).
                   -kwfunc (optional): lambda function which receives the coordinates of two 
                            neighbouring time-frequency bins and computes the weight value at
                            the second bin given its distance from the first bin. The weight
                            values fall in the interval [0,1]. Default: all ones over the 
                            neighbourhood (binary kernel).
                          
    AlgParams: Numpy array of length 2, containing algorithm parameters. The first element is
               the number of components or equivalently the rank of the mixture PSD, K, (default: 10),
               and the second element the compression exponent gamma (default: 1).                      
    
    Numit: (optional) number of iterations of the backfitting algorithm - default: 1     
                   
    SpecParams: (optional) structured containing spectrogram parameters including:
                         - windowtype (default is Hamming)
                         - windowlength (default is 60 ms)
                         - overlap_samples in [0,windowlength] (default is widowlength/2)
                         - num_fft_bins (default is windowlength)
                         - makeplot in {0,1} (default is 0)
                         - fmaxplot in Hz (default is fs/2)
                         example: 
                         SpecParams=np.zeros(1,dtype=[('windowtype','|S1'),
                                                      ('windowlength',int),
                                                      ('overlap_samples',int),
                                                      ('num_fft_bins',int),
                                                      ('makeplot',int),
                                                      ('fmaxplot',float)])
                         SpecParams['windowlength']=1024                             
    
    FullKernel: (optional) binary input which determines the method used for median filtering.
               If the kernel has a limited support and the same shape over all time-freq. bins,
               then instead of the full kernel method a sliding window can be used in the median 
               filtering stage in order to make the algorithm run faster (linear computational
               complexity). The default value is False.
               A True value means implementing the case where the similarity measure is 
               computed for all possible combinations of TF bins, resulting in quadratic 
               computational complexity, and hence longer running time.  
        
    Outputs:
    shat: a Ls by I by J Numpy array containing J time-domain source images on I channels   
    fhat: a LF by LT by J Numpy array containing J power spectral dencities      
    """

    # Step (1): 
    # Load the audio mixture from the input path
    if len(Inputfile) == 2:
        Mixture = AudioSignal(audiosig=Inputfile[0], fs=Inputfile[1])
    elif len(Inputfile) == 3:
        Mixture = AudioSignal(file_name=Inputfile[0], siglen=Inputfile[1], sigstart=Inputfile[2])

    numcomp, gamma = AlgParams

    if len(SpecParams) != 0:
        for i in range(0, len(SpecParams.dtype)):
            nameTemp = SpecParams.dtype.names[i]
            valTemp = SpecParams[0][i]
            valTemp = str(valTemp)
            exec ('Mixture.' + nameTemp + '=' + valTemp)

    x, tvec = np.array([Mixture.x, Mixture.time])  # time-domain channel mixtures
    X, Px, Fvec, Tvec = Mixture.do_STFT()  # stft and PSD of the channel mixtures

    I = Mixture.numCh  # number of channel mixtures
    J = len(SourceKernels)  # number of sources
    LF = np.size(Fvec)  # length of the frequency vector
    LT = np.size(Tvec)  # length of the timeframe vector

    F_ind = np.arange(LF)  # frequency bin indices
    T_ind = np.arange(LT)  # time frame indices
    Tmesh, Fmesh = np.meshgrid(T_ind, F_ind)  # grid of time-freq indices for the median filtering step
    TFcoords = np.mat(np.zeros((LF * LT, 2), dtype=int))  # all time-freq index combinations
    TFcoords[:, 0] = np.mat(np.asarray(Fmesh.T).reshape(-1)).T
    TFcoords[:, 1] = np.mat(np.asarray(Tmesh.T).reshape(-1)).T

    # Generate source kernels:
    Kj = []
    for ns in range(0, J):
        SKj = SourceKernels[ns]
        if len(SKj) < 2:
            raise Exception('The information required for generating source kernels is insufficient.'
                            ' Each sub-list in SourceKernels must contain at least two elements.')
        KTYPE = SKj[0]
        if KTYPE != 'userdef' and len(SKj) == 2:
            Kj.append(Kernel(Type=SKj[0], ParamVal=SKj[1]))
        elif KTYPE != 'userdef' and len(SKj) == 3:
            Kj.append(Kernel(Type=SKj[0], ParamVal=SKj[1]), Wfunc=SKj[2])
        elif KTYPE == 'userdef' and len(SKj) == 2:
            Kj.append(Kernel(Type=SKj[0], Nhood=SKj[1]))
        elif KTYPE == 'userdef' and len(SKj) == 3:
            Kj.append(Kernel(Type=SKj[0], Nhood=SKj[1]), Wfunc=SKj[2])


    # Step (2): initialization
    # Initialize the PSDs with average mixture PSD and the spatial covarince matricies
    # with identity matrices

    X = np.reshape(X.T, (LF * LT, I))  # reshape the stft tensor into I vectors
    if I > 1:
        MeanPSD = np.mean(Px, axis=2) / (I * J)
    else:
        MeanPSD = Px / (J)

    U = []
    V = []
    Utemp, Vtemp = randSVD(MeanPSD ** gamma, numcomp, 'compact')[0:3:2]  # compute the compact form of randomized SVD
    del MeanPSD

    numcomp = Utemp.shape[1]  # update the number of components in case the rank of the mixture PSD turns out
    # to be less than the input value or the default value for K

    for ns in range(0, J):
        U.append(Utemp)
        V.append(Vtemp)
    del Utemp, Vtemp

    Rj = 1j * np.zeros((1, I * I, J))
    for j in range(0, J):
        Rj[0, :, j] = np.reshape(np.eye(I), (1, I * I))
    Rj = np.tile(Rj, (LF * LT, 1, 1))

    ### Kernel Backfitting ###
    S = 1j * np.zeros((LF * LT, I, J))
    for n in range(0, Numit):

        # Step (3-a):
        # compute the inverse_mask term: [sum_j' (pgamma_j')^(1/gamma) R_j']^-1
        SumPR = np.zeros((LF * LT, I * I), dtype='single')
        for ns in range(0, J):
            Pj_gamma = np.abs(np.dot(U[ns], np.conj(V[ns].T)) ** (1 / gamma))
            Pj_reshape = np.reshape(Pj_gamma.T, (LF * LT, 1))
            Pj_tile = np.tile(Pj_reshape, (1, I * I))
            SumPR += Pj_tile * Rj[:, :, ns]
            del Pj_gamma, Pj_reshape, Pj_tile
        SumPR.shape = (LF * LT, I, I)
        SumPR += 1e-16 * np.random.randn(LF * LT, I, I)

        InvSumPR = np.zeros((LF * LT, I, I), dtype='single')
        if I == 1:
            InvSumPR = 1 / SumPR
        elif I == 2:
            InvDet = 1 / (SumPR[:, 0, 0] * SumPR[1, 1] - SumPR[0, 1] * SumPR[1, 0])
            InvSumPR[:, 0, 0] = InvDet * SumPR[:, 1, 1]
            InvSumPR[:, 0, 1] = -InvDet * SumPR[:, 0, 1]
            InvSumPR[:, 1, 0] = -InvDet * SumPR[:, 1, 0]
            InvSumPR[:, 1, 1] = InvDet * SumPR[:, 0, 0]
        else:
            InvSumPR = np.linalg.inv(SumPR)
        InvSumPR.shape = (LF * LT, I * I)
        del SumPR

        # compute sources, update PSDs and covariance matrices 
        for ns in range(0, J):
            Pj_gamma = np.abs(np.dot(U[ns], np.conj(V[ns].T)) ** (1 / gamma))
            Pj_reshape = np.reshape(Pj_gamma.T, (LF * LT, 1))
            Pj_tile = np.tile(Pj_reshape, (1, I * I))
            PRinvsum = Pj_tile * Rj[:, :, ns] * InvSumPR
            del Pj_gamma, Pj_reshape, Pj_tile

            Stemp = 1j * np.zeros((LF * LT, I))
            for nch in range(0, I):
                PRtemp = PRinvsum[:, nch * I:nch * I + 2]
                Stemp[:, nch] = np.sum(PRtemp * X, axis=1)
            del PRinvsum, PRtemp
            S[:, :, ns] = Stemp

            # Step (3-b):
            Cj = np.repeat(Stemp, I, axis=1) * np.tile(np.conj(Stemp), (1, I))

            # Step (3-c):
            Cj_reshape = np.reshape(Cj, (LF * LT, I, I))
            Cj_trace = np.mat(np.matrix.trace(Cj_reshape.T)).T
            MeanCj = Cj / np.tile(Cj_trace, (1, I * I))
            MeanCj_reshape = np.reshape(np.array(MeanCj), (LF, LT, I * I), order='F')
            Rj[:, :, ns] = np.tile(np.sum(MeanCj_reshape, axis=1), (LT, 1))

            # Step (3-d):
            Rj_reshape = np.reshape(Rj[:, :, ns], (LF * LT, I, I))
            Rj_reshape += 1e-16 * np.random.randn(LF * LT, I, I)

            InvRj = np.zeros((LF * LT, I, I), dtype='single')
            if I == 1:
                InvRj = 1 / Rj_reshape
            elif I == 2:
                InvDetR = 1 / (Rj_reshape[:, 0, 0] * Rj_reshape[1, 1] - Rj_reshape[0, 1] * Rj_reshape[1, 0])
                InvRj[:, 0, 0] = InvDetR * Rj_reshape[:, 1, 1]
                InvRj[:, 0, 1] = -InvDetR * Rj_reshape[:, 0, 1]
                InvRj[:, 1, 0] = -InvDetR * Rj_reshape[:, 1, 0]
                InvRj[:, 1, 1] = InvDetR * Rj_reshape[:, 0, 0]
            else:
                InvRj = np.linalg.inv(Rj_reshape)
            InvRj.shape = (LF * LT, I * I)

            InvRjCj = np.reshape(InvRj * Cj, (LF * LT, I, I))
            zj = np.real(np.matrix.trace(InvRjCj.T) / I)
            zj = np.mat(zj)

            # Step (3-e):
            # Median filter the estimated PSDs

            # start_time = time.clock()
            Ktemp = Kj[ns]  # Kernel corresponding to the source #j

            if not FullKernel:  # kernel defined as a sliding window (faster method)
                centerF = (LF - np.mod(LF, 2)) / 2  # middle freq. bin
                centerT = (LT - np.mod(LT, 2)) / 2  # middle time frame
                centerTF = np.mat([centerF, centerT])  # middle time-freq. bin
                KWin = Ktemp.sim(centerTF, TFcoords)  # sliding kernel window
                KWin_reshape = np.reshape(KWin, (LT, LF)).T
                NZ = np.nonzero(KWin_reshape)  # range of element numbers of central nonzero elements
                KWin_shrink = KWin_reshape[NZ[0].min():NZ[0].max() + 1,
                              NZ[1].min():NZ[1].max() + 1]  # extract the central nonzero part
                ZMed = scipy.ndimage.filters.median_filter(np.reshape(zj, (LT, LF)).T,
                                                           footprint=KWin_shrink)  # median filter
                U[ns], V[ns] = randSVD(ZMed ** (1 / gamma), numcomp, 'compact')[0:3:2]
                del ZMed, zj

            else:  # full kernel method (more general but slower approach)
                ZMed = np.zeros((LF * LT, 1), dtype='single')
                for ft in range(0, LF * LT):
                    simTemp = Ktemp.sim(TFcoords[ft, :], TFcoords)
                    NhoodTemp = np.nonzero(simTemp)
                    zjNhood = np.multiply(zj[NhoodTemp], simTemp[NhoodTemp])
                    ZMed[ft, 0] = np.median(np.array(zjNhood))
                ZMed = np.reshape(ZMed, (LT, LF)).T
                U[ns], V[ns] = randSVD(ZMed ** (1 / gamma), numcomp, 'compact')[0:3:2]
                del ZMed, zj

                # print time.clock() - start_time, "seconds"

    # print time.clock() - start_time, "seconds"


    # Reshape the PSDs
    phat = np.zeros((LF, LT, J))
    for ns in range(0, J):
        phat[:, :, ns] = np.abs(np.dot(U[ns], np.conj(V[ns].T)) ** (1 / gamma))

    # Reshape the spectrograms
    Shat = 1j * np.zeros((LF, LT, I, J), dtype='single')  # estimated source STFTs
    for ns in range(0, J):
        for nch in range(0, I):
            Shat[:, :, nch, ns] = np.reshape(S[:, nch, ns], (LT, LF)).T


    # Compute the inverse_mask stft of the estimated sources
    shat = np.zeros((x.shape[0], I, J))
    sigTemp = AudioSignal()
    sigTemp.windowtype = Mixture.windowtype
    sigTemp.windowlength = Mixture.windowlength
    sigTemp.overlap_samples = Mixture.overlap_samples
    sigTemp.numCh = I
    for ns in range(0, J):
        sigTemp.X = Shat[:, :, :, ns]
        shat[:, :, ns] = sigTemp.istft()[0][0:x.shape[0]]

    return shat, phat


def randSVD(A, K, mode='normal'):
    """
    The function randSVD implements the randomized computation of truncated SVD
    of K components over a m by n matrix A.
    Inputs:
    A: Numpy array (m by n) 
    K: number of components
    mode: one of three cases
         - 'normal' (default): S is a K by K diagonal matrix
         - 'diagonal': S is the K by 1 vector containing the singular values
         - 'compact': U and V are both multiplied by sqrt(S), and S is set to 1.
     
    Outputs: 
    U: Numpy array (m by K) containing basis vectors in C^m
    S: Numpy array (K by K) containing singular values
    V: Numpy array (n by K) containing basis vectors in C^n
    """

    m, n = np.shape(A)
    #  Step 1: generate a random nx2K Gassian iid matrix Omega
    Omega = np.random.randn(n, np.min([2 * K, n]))
    # Step 2: form Y=A*Omega
    Y = np.dot(A, Omega)
    # Step 3: compute an orthonormal basis Q for the range of Y
    Q = scipy.linalg.orth(Y)
    # Step 4: form B=Q.T*A
    B = np.dot(np.conj(Q.T), A)
    # Step 5: compute svd of B
    Utilde, S, V = np.linalg.svd(B, full_matrices=False)
    # Step 6: form U=Q*Utilde
    U = np.dot(Q, Utilde)
    # Step 7: update the # of components and matrix sizes
    K = np.min(np.array([K, np.shape(U)[1]]))
    U = U[:, 0:K]
    S = np.diag(S[0:K])
    V = V.T[:, 0:K]

    if mode == 'diagonal':
        S = np.diag(S)
    elif mode == 'compact':
        sqrtS = np.diag(np.sqrt(np.diag(S)))
        U = np.dot(U, sqrtS)
        V = np.dot(V, sqrtS)
        S = np.eye(K)

    return U, S, V


class Kernel:
    """
    The class Kernel defines the properties of the time-freq proximity kernel. The weight values of 
    the proximity kernel over time-frequecy bins that are considered as neighbours are given
    by a pre-defined or a user-defined function. The value of the proximity kernel is zero over
    time-frequency bins outside the neighbourhood.
    
    Properties:
    
    -kType: (string) determines whether the kernel is one of the pre-defined kernel types 
             or a user-defined lambda function. 
             Predefined choices are: 'cross','horizontal','vertical','periodic'
             To define a new kernel type, kType should be set to: 'userdef'
             
    -kParamVal: a Numpy matrix containing the numerical values of the kernel parameters. If any
             of the pre-defined kernel type is selected, the parameter values should be provided 
             through kParamVal. Parameters corresponding to the pre-defined kernels are:
             Cross: (neighbourhood width along the freq. axis in # of freq. bins, neighbour width
                     along the time axis in # of time frames)
             Vertical: (neighbourhood width along the freq. axis in # of freq. bins)
             Horizontal: (neighbourhood width along the time axis in # of time frames)
             Periodic: (period in # of time frames,# of periods along the time axis) 
                        
             Note: neighbourhood width is measured in only one direction, e.g. only to the
                   right of a time-freq bin in the case of a horizontal kernel, so the whole
                   length of the neighbourhood would be twice the specified width.
             
    -kNhood: logical lambda funcion which receives the coordinates of two time-frequency
             bins and determines whether they are neighbours (outputs TRUE if neighbour).
             
    -kWfunc: lambda function which receives the coordinates of two time-frequency bins that are
             considered neighbours by kNhood and computes the weight value at the second bin given 
             its distance from the first bin. The weight values fall in the interval [0,1] with 
             1 indicating zero-distance or equivalently perfect similarity. 
             Default: all ones over the neighbourhood (binary kernel)
    
    EXAMPLE: 
    
    FF,TT=np.meshgrid(np.arange(5),np.arange(7))
    TFcoords1=np.mat('2,3')
    TFcoords2=np.mat(np.zeros((35,2)))
    TFcoords2[:,0]=np.mat(np.asarray(FF.T).reshape(-1)).T
    TFcoords2[:,1]=np.mat(np.asarray(TT.T).reshape(-1)).T

    W=lambda TFcoords1,TFcoords2: np.exp(-(TFcoords1-TFcoords2)*(TFcoords1-TFcoords2).T)
    k_cross=Kernel('cross',np.mat([3,2]),W)
    simVal_cross=np.reshape(k_cross.sim(TFcoords1,TFcoords2),(5,7))
                      
    """

    def __init__(self, Type='', ParamVal=np.mat([]), Nhood=None, Wfunc=None):

        """
        Inputs:
        Type: (string) determines whether the kernel is one of the pre-defined kernel types 
             or a user-defined lambda function. 
             Predefined choices are: 'cross','horizontal','vertical','periodic','harmonic'
             To define a new kernel type, kType should be set to: 'userdef'
             
        ParamVal: a Numpy matrix containing the numerical values of the kernel parameters. If any
             of the pre-defined kernel type is selected, the parameter values should be provided 
             through kParamVal. Parameters corresponding to the pre-defined kernels are:
             Cross: (neighbourhood width along the freq. axis in # of freq. bins, neighbour width
                     along the time axis in # of time frames)
             Vertical: (neighbourhood width along the freq. axis in # of freq. bins)
             Horizontal: (neighbourhood width along the time axis in # of time frames)
             Periodic: (period in # of time frames,# of periods along the time axis) 
             Harmonic: (period in # of freq. bins, # of periods along the freq. axis)
             
        Nhood: logical lambda funcion which receives the coordinates of two time-frequency
             bins and determines whether they are neighbours (outputs TRUE if neighbour).
             
        Wfunc: lambda function which receives the coordinates of two time-frequency bins that are
             considered neighbours by kNhood and computes the weight value at the second bin given 
             its distance from the first bin. The weight values fall in the interval [0,1] with 
             1 indicating zero-distance or equivalently perfect similarity. 
             Default: all ones over the neighbourhood (binary kernel)
        """

        if Type == 'userdef' and (Nhood is None):
            raise ValueError('Kernel type is userdef but the kernel neighbourhood is not defined.')

        # kernel properties
        self.kType = Type  # default: no pre-defined kernel selected
        self.kParamVal = ParamVal
        self.kNhood = Nhood
        self.kWfunc = Wfunc

        if self.kNhood is None:
            self.kNhood = lambda TFcoords1, TFcoords2: (
                TFcoords1 == TFcoords2).all()  # default: neighnourhood includes only the centeral bin
        if self.kWfunc is None:
            self.kWfunc = lambda TFcoords1, TFcoords2: self.kNhood(TFcoords1, TFcoords2)  # default: binary kernel

        if Type in ['cross', 'vertical', 'horizontal', 'periodic', 'harmonic']:
            self.gen_predef_kernel()

    def gen_predef_kernel(self):
        """
        generates the pre-defined kernel object given the parameters
        """

        Type = self.kType
        ParamVal = self.kParamVal

        if np.size(ParamVal) == 0:
            raise ValueError('Kernel parameter values are not specified.')

        if Type == 'cross':

            Df = ParamVal[0, 0]
            Dt = ParamVal[0, 1]
            self.kNhood = lambda TFcoords1, TFcoords2: np.logical_or(np.logical_and((np.tile(TFcoords1[:, 0], (
            1, TFcoords2.shape[0])) == np.tile(TFcoords2[:, 0].T, (TFcoords1.shape[0], 1))),
                                                                                    (np.abs(np.tile(TFcoords1[:, 1], (
                                                                                    1, TFcoords2.shape[0])) - np.tile(
                                                                                        TFcoords2[:, 1].T, (
                                                                                        TFcoords1.shape[0], 1))) < Dt)),
                                                                     np.logical_and((np.tile(TFcoords1[:, 1], (
                                                                     1, TFcoords2.shape[0])) == np.tile(
                                                                         TFcoords2[:, 1].T, (TFcoords1.shape[0], 1))),
                                                                                    (np.abs(np.tile(TFcoords1[:, 0], (
                                                                                    1, TFcoords2.shape[0])) - np.tile(
                                                                                        TFcoords2[:, 0].T, (
                                                                                        TFcoords1.shape[0], 1))) < Df)))
            self.kParamVal = ParamVal

        elif Type == 'vertical':

            Df = ParamVal[0, 0]
            self.kNhood = lambda TFcoords1, TFcoords2: np.logical_and((np.tile(TFcoords1[:, 1],
                                                                               (1, TFcoords2.shape[0])) == np.tile(
                TFcoords2[:, 1].T, (TFcoords1.shape[0], 1))),
                                                                      (np.abs(np.tile(TFcoords1[:, 0], (
                                                                      1, TFcoords2.shape[0])) - np.tile(
                                                                          TFcoords2[:, 0].T,
                                                                          (TFcoords1.shape[0], 1))) < Df))
            self.kParamVal = ParamVal

        elif Type == 'horizontal':

            Dt = ParamVal[0, 0]
            self.kNhood = lambda TFcoords1, TFcoords2: np.logical_and((np.tile(TFcoords1[:, 0],
                                                                               (1, TFcoords2.shape[0])) == np.tile(
                TFcoords2[:, 0].T, (TFcoords1.shape[0], 1))),
                                                                      (np.abs(np.tile(TFcoords1[:, 1], (
                                                                      1, TFcoords2.shape[0])) - np.tile(
                                                                          TFcoords2[:, 1].T,
                                                                          (TFcoords1.shape[0], 1))) < Dt))
            self.kParamVal = ParamVal

        elif Type == 'periodic':

            P = ParamVal[0, 0]
            Dt = ParamVal[0, 1] * P + 1
            self.kNhood = lambda TFcoords1, TFcoords2: np.logical_and(np.logical_and((np.tile(TFcoords1[:, 0], (
            1, TFcoords2.shape[0])) == np.tile(TFcoords2[:, 0].T, (TFcoords1.shape[0], 1))),
                                                                                     (np.abs(np.tile(TFcoords1[:, 1], (
                                                                                     1, TFcoords2.shape[0])) - np.tile(
                                                                                         TFcoords2[:, 1].T, (
                                                                                         TFcoords1.shape[0],
                                                                                         1))) < Dt)),
                                                                      (np.mod(np.tile(TFcoords1[:, 1], (
                                                                      1, TFcoords2.shape[0])) - np.tile(
                                                                          TFcoords2[:, 1].T, (TFcoords1.shape[0], 1)),
                                                                              P) == 0))
            self.kParamVal = ParamVal

        elif Type == 'harmonic':

            P = ParamVal[0, 0]
            Df = ParamVal[0, 1] * P + 1
            self.kNhood = lambda TFcoords1, TFcoords2: np.logical_and(np.logical_and((np.tile(TFcoords1[:, 1], (
            1, TFcoords2.shape[0])) == np.tile(TFcoords2[:, 1].T, (TFcoords1.shape[0], 1))),
                                                                                     (np.abs(np.tile(TFcoords1[:, 0], (
                                                                                     1, TFcoords2.shape[0])) - np.tile(
                                                                                         TFcoords2[:, 0].T, (
                                                                                         TFcoords1.shape[0],
                                                                                         1))) < Df)),
                                                                      (np.mod(np.tile(TFcoords1[:, 0], (
                                                                      1, TFcoords2.shape[0])) - np.tile(
                                                                          TFcoords2[:, 0].T, (TFcoords1.shape[0], 1)),
                                                                              P) == 0))
            self.kParamVal = ParamVal

    def sim(self, TFcoords1, TFcoords2):
        """
        Measures the similarity between a series of new time-freq points and the kernel central point.

        Inputs:
        TFcoords1: N1 by 2 Numpy matrix containing coordinates of N1 time-frequency bins.
                   Each row contains the coordinates of a single bin.
        TFcoords2: N2 by 2 Numpy matrix containing coordinates of N2 time-frequency bins.

        Output:
        simVal: N1 by N2 Numby matrix of similarity values. Similarity values fall in the interval [0,1].
                The value of the (i,j) element in simVal determines the amountof similarity (or closeness)
                between the i-th time-frequency bin in TFcoords1 and j-th time-frequency bin in TFcoords2.
        """

        # update the kernel properties if changed to predefined
        if self.kType in ['cross', 'vertical', 'horizontal', 'periodic']:
            self.gen_predef_kernel()

        Nhood_vec = self.kNhood(TFcoords1, TFcoords2)
        Wfunc_vec = self.kWfunc(TFcoords1, TFcoords2)
        simVal = np.multiply(Nhood_vec, Wfunc_vec).astype(np.float32)

        return simVal
