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
from scipy.fftpack import fft
from scipy.fftpack import ifft
from f_stft import f_stft
from f_istft import f_istft
import matplotlib.pyplot as plt
plt.interactive('True')


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
    
    
    
    
    
    
    
    
    
    