"""
In this module the Robust Principle Component Analysis (RPCA) method is used to 
perform source (background/foreground) separation. 
Assuming that a musical signal is composed of a repeating background (low rank) 
and a sparse foreground, the RPCA decomposition can be used to decompose the 
spectrogram of the mixture and hence separate the background from the foreground.

References:
[1] Huang, Po-Sen, et al. "Singing-voice separation from monaural recordings 
    using robust principal component analysis." Acoustics, Speech and Signal 
    Processing (ICASSP), 2012 IEEE International Conference on. IEEE, 2012.
    
[2] Candes, Emmanuel J., et al. "Robust principal component analysis?." 
    Journal of the ACM (JACM) 58.3 (2011): 11.    
    
Required packages:
1. Numpy
2. Scipy
3. Matplotlib

Required modules:    
1. f_stft
2. f_istft

"""

import numpy as np 
from f_stft import f_stft
from f_istft import f_istft
import matplotlib.pyplot as plt
plt.interactive('True')

def rpca_ss(x,fs,specparam=None):
    """
    The function rpca_ss uses the RPCA method to decompose the power spectral 
    density of a mixture into its background (assumed to be low-rank) and 
    foreground (assumed to be sparse). A binary mask is formed using the results
    of RPCA and applied to the mixture to perform background/foreground separation.
    
    x: M by N numpy array containing M channel mixtures, each of length N time 
       samples
    fs: sampling frequency of the audio signal
    specparam(optional): list containing STFT parameters including in order, 
                         the window length, window type, overlap in # of samples, 
                         and # of fft points.
                         default: window length: 40 ms
                         window type: Hamming
                         overlap: window length/2 (50%)
                         nfft: window length (no zero padding)
                         
    Output:
    y_f: M by N numpy array containing the foreground (sparse component), M 
         source images, each of length N time samples
    y_b: M by N numpy array containing the background (repeating/low-rak component),
         M source images, each of length N time samples    
         
    EXAMPLE:
     
    """
    
    # use the default range of repeating period and default STFT parameter values 
    # if not specified
    if specparam is None:
       winlength=int(2**(np.ceil(np.log2(0.04*fs)))) # next power 2 of 40ms*fs
       specparam = [winlength,'Hamming',winlength/2,winlength]
   
   # STFT parameters      
    L,win,ovp,nfft=specparam 
    
    # compute the spectrograms of all channels
    M,N = np.shape(x)
    X=f_stft(np.array(x[0,:],ndmin=2),L,win,ovp,fs,nfft,0)[0]
    for i in range(1,M):
         Sx = f_stft(np.mat(x[i,:]),L,win,ovp,fs,nfft,0)[0]
         X=np.dstack([X,Sx])
    V=np.abs(X)  
    if M==1: 
        X=X[:,:,np.newaxis]
        V=V[:,:,np.newaxis]
        
    # compute the masks (using rpca)    
    
    Bmask=np.zeros((V.shape)) 
    Fmask=np.zeros((V.shape))
        
    
    return


def rpca(M,delta=1e-7):
    """
    The function rpca implements the Robust Principle Component Analysis (RPCA) 
    method to decompose a matrix into its low rank and sparse components. 
    The augmented Lagrange multiplier (ALM) algorithm is used to solve the 
    optimization problem that outputs the decomposed parts. The optimization 
    problem can be stated as: min ||L||* + lambda ||S||1  s.t. L+S = M
    
    Inputs:
    M: Numpy n by m array containing the original matrix elements
    delta: stopping criterion
    
    Outputs:
    L: Numpy n by m array containing the elements of the low rank part
    S: Numpy n by m array containing the elements of the sparse part 
    
    """
    
    # compute the dimensions of the input matrix, M, and initialize the low-rank 
    # matrix,L, sparse matrix,S, and the matrix of residuals, Y
    n,m=np.shape(M)
    L=np.zeros((n,m))
    S=np.zeros((n,m))
    Y=np.zeros((n,m))
    
    # compute the (rule of thumb) velues of the lagrange multiplyer and the 
    # svd-threshold 
    Lambda = 1/np.sqrt(np.max([n,m]))
    mu = (n*m)/(4. * np.linalg.norm(M,ord=1))        
               
    # initialize the error value (loop condition)
    Etemp = np.linalg.norm(M - L - S,ord='fro') / np.linalg.norm(M,ord='fro')
    Error=np.array([Etemp],ndmin=2)
    
    while Etemp>delta:
        L = svd_thr(M - S + Y/mu,1/mu)
        S = shrink(M - L + Y/mu, Lambda/mu)
        Y = Y + mu*(M - L - S)
        Etemp = np.linalg.norm(M - L - S,ord='fro') / np.linalg.norm(M,ord='fro')
        Error=np.hstack([Error,np.array(Etemp,ndmin=2)])
    
    return L,S,Error
    
 
def shrink(X,tau):
    """
    The function shrink applies the shrinkage operator to the input matrix and
    computes the output according to: S_tau = shrink(Xij) = sgn(Xij)max(abs(Xij)-tau,0).
    Note: ij index denotes element-wise operation
    
    Inputs:
    S: Numpy n by m array 
    tau: shrinkage parameter 
    
    Output:
    S_tau: Numpy n by m array containing the elements of the shrinked matrix
    """
    
    S_tau = np.sign(X)*np.maximum(np.abs(X)-tau,0)
        
    return S_tau
    
    
def svd_thr(X,tau):
    """
    The function svd_thr applies the singular value thresholding operator to 
    the input matrix and computes the output according to: 
    D_tau = svd_thr(X) = U S_tau(sigma) V* where X = U sigma V* is any singular 
    value decomposition. 
    
    Inputs: 
    X: Numpu n by m array
    tau: shrinkage parameter 
    
    Output: 
    D_tau: Numpy n by m array containing elements of the matrix composed of 
           thresholded singular values
    """    
    
    U,sigma,V = np.linalg.svd(X, full_matrices=False)
    S_tau = shrink(sigma,tau)
    D_tau = np.dot(U, np.dot(np.diag(S_tau), V))
        
    return D_tau
    
    
    
    
    
    
    
    
    
    