# -*- coding: utf-8 -*-

"""
This function computes the on-sided STFT of a signal 

 Inputs:
X: signal, column vector
L: length of the window
win: window type, strignt (Rectangular, Hamming, Hanning, Blackman)
ovp: number of overlapping samples
nfft: number of freq. samples per time step
fs: sampling rate of the signal
mkplot: binary input (1 for make plot)

Outputs:
S: short-time Fourier Transform (complex)
F: frequency vector
T: time vector
P: PSD of the signal

* Note: windowing and fft will be performed row-wise so that the code to run faster
"""

def f_stft(X,L,win,ovp,nfft,fs,mkplot):
    
    import numpy as np 
    import matplotlib.pyplot as plt
    from scipy.fftpack import fft
    
    # split data into blocks
    Hop=int(L-ovp)
    N=len(X)
    Xz=np.zeros((1,N))
    Xz[0,:]=X; X=Xz;
    
    # zero-pad the vector at the beginning and end to reduce the window tapering effect
    if np.mod(L,2)==0:
        zp1=L/2
    else:
        zp1=(L-1)/2
        
    X=np.hstack([np.zeros((1,zp1)),X,np.zeros((1,zp1))])    
    N=N+2*zp1

    # zero pad if N-L is not an integer multiple of Hop
    rr=np.mod(N-L,Hop);
    if rr!=0:
        zp2=Hop-rr
        X=np.vstack([X,np.zeros((zp2,1))])
        N=len(X)
    else:
        zp2=0

    NumBlock=int(((N-L)/Hop)+1)
    
    # Generate samples of a normalized window
    W=np.zeros((1,L))
    if (win=='Rectangular'): W[0,:]=np.ones((L,1))
    elif (win=='Hamming'): W[0,:]=np.hamming(L)
    elif (win=='Hanning'): W[0,:]=np.hanning(L)
    elif (win=='Blackman'): W[0,:]=np.blackman(L)
        
    Wnorm2=np.dot(W,W.T)
    
    # Generate freq. vector
    nfft=int(2**np.ceil(np.log2(nfft)))
    F=(fs/2)*np.linspace(0,1,num=nfft/2+1)
    Lf=len(F)    
    
    # Take the fft of each block
    S=1j*np.zeros((NumBlock,Lf));  # row: time, col: freq. to increase speed
    P=np.zeros((NumBlock,Lf)); 
        
    for i in range(0,NumBlock):
        Xw=W*X[0,(i*Hop):(i*Hop+L)]
        XX=fft(Xw,n=nfft)        
        XX_trun=XX[0,0:Lf]
       
        S[i,:]=XX_trun
        P[i,:]=(1/float(fs))*((abs(S[i,:])**2)/float(Wnorm2))
    S=S.T;  P=P.T; # row: freq col: time to get conventional spectrogram orientation 
        
    Th=float(Hop)/float(fs)
    T=np.arange(0,(NumBlock)*Th,Th)

    Ls1,Ls2=np.shape(S)
    m1=np.floor(zp1/Hop)
    m2=np.ceil((zp1+zp2)/Hop)
    S=S[:,m1:Ls2-m2]
    P=P[:,m1:Ls2-m2]
    T=T[m1:Ls2-m2]
        
    # plot if asked
    if mkplot==1:
        SP=10*np.log10(np.abs(P))
        plt.figure(1)
        plt.imshow(SP)
        plt.gca().invert_yaxis()
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.ylim(0,100)
        plt.show()

    return S,P,F,T    
    

         
    
         
        
    
  


        