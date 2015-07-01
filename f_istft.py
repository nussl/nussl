"""
This function computes the inverse STFT of a spectrogram using 
Overlap Adition method

Inputs:
X: one-sided STFT (spectrogram) of the signal x
L: window length 
win(optional): window type, (string): 'Rectangular', 'Hamming', 'Hanning', 'Blackman'
   (default window type: Hamming)
ovp: overlap between adjacent windows in STFT analysis
nfft: number of freq. samples in (-pi,pi]
fs: sampling rate of the original signal x

Outputs:
t: vector of time values for the reconstructed signal
y: the reconstructed signal

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
  y,t = f_istft(S,L,win,ovp,nfft,fs)
  
Required packages:
1. Numpy
2. Scipy

"""

def f_istft(*arg):
   
    import numpy as np 
    from scipy.fftpack import ifft
    
    if len(arg)==5: 
        X,L,ovp,nfft,fs = arg[0:5]
        win='Hamming'
    elif len(arg)==6:
        X,L,win,ovp,nfft,fs = arg[0:6]
    
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
    
    for h in range(1,Nc):
        yh=np.real(ifft(X[:,h-1]))
        hh=int((h-1)*Hop)
        y[0,hh:hh+L]=y[0,hh:hh+L]+yh[0:L]
        
    c=sum(W)/Hop
    y=y[0,:]/c
    #y=y[0,:]
    t=np.arange(Ly)/float(fs)
    
    return y,t
        
        
        
        
        
    
    
    
    