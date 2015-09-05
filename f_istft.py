"""
This function computes the inverse STFT of a spectrogram using 
Overlap Adition method

Inputs:
X: one-sided STFT (spectrogram) of the signal x
L: window length (in # of samples)
win(optional): window type, (string): 'Rectangular', 'Hamming', 'Hanning', 'Blackman'
ovp: overlap between adjacent windows in STFT analysis
fs: sampling rate of the original signal x

Outputs:
t: Numpy array containing time values for the reconstructed signal
y: Numpy array containing the reconstructed signal

EXAMPLE:
# first generate the spectrogram of a signal and then reconstruct the spectrogram back 
to the time-domain (the spectrogram in real problems would go through further 
processing, which for simplicity is not shown here) 
  fs=16000
  t=np.mat(np.arange(fs+1)/float(fs))
  x=np.cos(2*np.pi*440*t)
  L=1024
  win='Hanning'
  ovp=0.5*L
  nfft=1024
  mkplot=1
  fmax=1000
  S,P,F,T = f_stft(x,L,win,ovp,nfft,fs,mkplot,fmax)
  y,t = f_istft(S,L,win,ovp,fs)
  
Required packages:
1. Numpy
2. Scipy

"""

import numpy as np 
from scipy.fftpack import ifft

def f_istft(X,L,win,ovp,fs):
   
    
    # Get spectrogram dimenstions and compute window hop size
    Nc=X.shape[1] # number of columns of X
    Hop=int(L-ovp)
    
    # Form the full spectrogram (-pi,pi]
    Xext=X[-2:0:-1,:];
    X_inv_conj=Xext.conj()
    X=np.vstack([X,X_inv_conj])
    
    # Generate samples of a normalized window
    if (win=='Rectangular'): W=np.ones((L,1))
    elif (win=='Hamming'): W=np.hamming(L)
    elif (win=='Hanning'): W=np.hanning(L)
    elif (win=='Blackman'): W=np.blackman(L)
            
    ## Reconstruction through OLA
    
    Ly=int((Nc-1)*Hop+L)
    y=np.zeros((1,Ly))
    
    for h in range(0,Nc):
        yh=np.real(ifft(X[:,h]))
        hh=int((h)*Hop)
        y[0,hh:hh+L]=y[0,hh:hh+L]+yh[0:L]
        
    c=sum(W)/Hop
    y=y[0,:]/c
    t=np.arange(Ly)/float(fs)
    
    return y,t
        
        
        
        
        
    
    
    
    