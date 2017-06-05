# -*- coding: utf-8 -*-
"""
Implements PROJET for separating spatial audio
as described in the paper:

@inproceedings{fitzgeraldPROJETa,
TITLE = {{PROJET - Spatial Audio Separation Using Projections}},
AUTHOR = {D. Fitzgerald and A. Liutkus and R. Badeau},
BOOKTITLE = {{41st International Conference on Acoustics, Speech and Signal Processing (ICASSP)}},
ADDRESS = {Shanghai, China},
PUBLISHER = {{IEEE}},
YEAR = {2016},
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


#importing stuff
import numpy as np
try:
    import numpy.core._dotblas
    print 'FAST BLAS'
except ImportError:
    print 'Slow BLAS'
from basics import wav,stft
import os
import warnings
warnings.filterwarnings("ignore")

eps = 1e-20


def multichannelGrid(I,L,sigma=1,normalize=True):
    pos = np.linspace(0,I-1,L)
    res = np.zeros((I,L))
    for i in range(I):
        res[i,...]=np.exp(-(pos-i)**2 / sigma**2)
    if normalize:
        res /= np.sqrt(np.sum(res**2,axis=0))
    return res

def complex_randn(shape):
    return np.random.randn(*shape)+1j*np.random.randn(*shape)

def orthMatrix(R):
    (I,L) = R.shape
    res = np.ones((L,I))
    res[:,-1] = - (R[-1,:]/np.squeeze(np.sum(R[:-1,:],axis=0))).T
    res /= np.sqrt(np.sum(res**2,axis=1))[...,None]
    return res

def separate(inputFilename, outputDir,J,L,M,nIter = 1000):
    """ performs separation using PROJET algorithm, as presented in the following paper:

    * inputFilename is the link to the wav file
    * outputDir is the folder where to write the results
    * J is the number of sources
    * L is the number of potential panning directions
    * M is the number of projections to use
    * nIter is the number of iterations

    """
    nfft = 4096
    overlap = 0.75 
    maxLength = np.inf

    
    #1) Loading data
    #----------------
    basenameFile = os.path.split(inputFilename)[1]
    (sig,fs) = wav.wavread(inputFilename, maxLength)
    fs = float(fs)
    
    print 'Launching STFT'
    hop = float(nfft)*(1.0-overlap)
    X = stft.stft(sig, nfft, hop, real=True, verbose=False).astype(np.complex64)
    (F,T,I) = X.shape 
    
    #initialize PSD and panning to random
    P = np.abs(np.random.randn(F*T,J),dtype='float32')+1
    Q = np.abs(np.random.randn(L,J),dtype='float32')+1
   
    
    #compute panning profiles
    #30 for regular gridding, the others as random
    panning_matrix =  np.concatenate((complex_randn((I, L-30)), multichannelGrid(I,30)),axis=1)
    panning_matrix /= np.sqrt(np.sum(np.abs(panning_matrix)**2,axis=0))[None,...]

    #compute projection matrix
    #5 for orthoganal to a regular gridding, the others as random
    projection_matrix =  np.concatenate((complex_randn((max(M-5,0), I)),orthMatrix(multichannelGrid(I,min(M,5)))))
    projection_matrix /= np.sqrt(np.sum(np.abs(projection_matrix)**2,axis=1))[...,None]
    
    #compute K matrix
    K = np.abs(np.dot(projection_matrix, panning_matrix)).astype(np.float32)
    
    #compute the projections and store their spectrograms and squared spectrograms
    C = np.reshape(np.tensordot(X,projection_matrix,axes=(2,1)),(F*T,M))
    V = np.abs(C).astype(np.float32)
    V2 = V**2
    C = [] #release memory
            
    #main iterations
    for iteration in range(nIter):
        print 'PROJET [%d/%d]'%(iteration+1,nIter)
        print '    updating P'
        sigma = np.dot(P,np.dot(Q.T,K.T))
        P *= np.dot(1.0/(sigma+eps),np.dot(K,Q))/(np.dot(3*sigma/(sigma**2+V2+eps),np.dot(K,Q)))

        #the following line is an optional trick that enforces orthogonality of the spectrograms.
        #P*=(100+P)/(100+np.sum(P,axis=1)[...,None])
        
        print '    updating Q'
        sigma = np.dot(P,np.dot(Q.T,K.T)).T
        Q *= np.dot(K.T,np.dot(1.0/(sigma+eps) ,P))/np.dot(K.T,np.dot(3*sigma/(sigma**2+V2.T+eps),P))
                
    #final separation
    print 'final separation...'
    recompose_matrix = np.linalg.pinv(projection_matrix) #IxM

    sigma = np.dot(P,np.dot(Q.T,K.T))
    C = np.dot(np.reshape(X,(F*T,I)),projection_matrix.T)
    
    for j in range(J):
        print '    source %d/%d'%(j+1,J)
        sigma_j = np.outer(P[:,j],np.dot(Q[:,j].T,K.T))
        yj = sigma_j/sigma*C
        yj = np.dot(yj,recompose_matrix.T)
        yj = np.reshape(yj,(F,T,I))
        yj = stft.istft(yj,1,hop,real=True,shape=sig.shape).astype(np.float32)
        sourceFilename = os.path.join(outputDir,'%s_M=%d_source_%d'%(basenameFile,M,j+1)+'.wav')
        wav.wavwrite(yj, fs, sourceFilename,verbose=False)

