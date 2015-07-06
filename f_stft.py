"""
This function computes the one-sided STFT of a signal 

Inputs:
X: signal, row vector
L: length of one window (in # of samples)
win: window type, string ('Rectangular', 'Hamming', 'Hanning', 'Blackman')
ovp: number of overlapping samples between adjacent windows
nfft: min number of desired freq. samples in (-pi,pi]. MUST be >= L. *NOTE* If 
this is not a power of 2, then it will automatically zero-pad up to the next power 
of 2. IE if you put 257 here, it will pad up to 512.
fs: sampling rate of the signal
mkplot: binary input (1 for show plot)
fmax(optional): maximum frequency shown on the spectrogram if mkplot is 1

Outputs:
S: 2D numpy matrix containing the short-time Fourier transform of the signal (complex)
P: 2D numpy matrix containing the PSD of the signal
F: frequency vector
T: time vector

* Note: windowing and fft will be performed row-wise so that the code runs faster

EXAMPLE:
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
 
Required packages:
1. Numpy
2. Scipy
3. Matplotlib

"""

def f_stft(*arg):
    
    import numpy as np 
    import matplotlib.pyplot as plt
    from scipy.fftpack import fft
    
    if len(arg)==7 and arg[6]==0: 
        X, L, win, ovp, nfft, fs, mkplot = arg[0:7]
    elif len(arg)==7 and arg[6]==1: 
        X, L, win, ovp, nfft, fs, mkplot = arg[0:7]
        fmax = fs/2
    elif len(arg)==8: 
        X, L, win, ovp, nfft, fs, mkplot = arg[0:7]
        fmax=arg[7]
    
    # split data into blocks (make sure X is a row vector)
    if np.shape(X)[0]!=1:
        raise ValueError('X must be a row vector')
    elif nfft<L:
        raise ValueError('nfft must be greater or equal the window length (L)!')
           
    Hop=int(L-ovp)
    N=np.shape(X)[1]
    
    
    # zero-pad the vector at the beginning and end to reduce the window tapering effect
    if np.mod(L,2)==0:
        zp1=L/2
    else:
        zp1=(L-1)/2
        
    X=np.hstack([np.zeros((1,zp1)),X,np.zeros((1,zp1))])    
    N=N+2*zp1

    # zero pad if N-L is not an integer multiple of Hop
    rr=np.mod(N-L,Hop)
    if rr!=0:
        zp2=Hop-rr
        X=np.hstack([X,np.zeros((1,zp2))])
        N=np.shape(X)[1]
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
        Xw=np.multiply(W,X[0,(i*Hop):(i*Hop+L)])
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
        
    # plot if specified
    if mkplot==1:
        
        TT=np.tile(T,(len(F),1))
        FF=np.tile(F.T,(len(T),1)).T
        SP=10*np.log10(np.abs(P))
        plt.pcolormesh(TT,FF,SP)
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.xlim(0,T[-1])
        plt.ylim(0,fmax)
        plt.show()

    return S,P,F,T    
    

         
    
         
        
    
  


        