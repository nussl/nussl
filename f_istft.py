## This function computes the inverse STFT of a spectrogram using 
# Overlap Adition method
#
# Inputs:
# X: one-sided STFT (spectrogram) of the signal x
# L: window length (default window type: Hamming)
# win: window type, (string): Rectangular, Hamming, Hanning, Blackman
# ovp: overlap between adjacent windows in STFT analysis
# nfft: number of freq. samples in (-pi,pi]
# fs: sampling rate of the original signal x
#
# Outputs:
# t: vector of time values for the reconstructed signal
# y: the reconstructed signal

def f_istft(X,L,win,ovp,nfft,fs):
    
    import numpy as np 
    from scipy.fftpack import ifft
    
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
    y=y/c
    t=np.arange(Ly)/float(fs)
    
    return y,t
        
        
        
        
        
    
    
    
    