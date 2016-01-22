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

def rpca_ss(x,fs,specparam=None,mask=False,maskgain=0):
    """
    The function rpca_ss uses the RPCA method to decompose the power spectral 
    density of a mixture into its background (assumed to be low-rank) and 
    foreground (assumed to be sparse). A binary mask is formed using the results
    of RPCA and applied to the mixture to perform background/foreground separation.
    
    x: M by N numpy array containing M channel mixtures, each of length N time 
       samples
    fs: sampling frequency of the audio signal
    specparam(optional): list containing do_STFT parameters including in order,
                         the window length, window type, overlap in # of samples, 
                         and # of fft points.
                         default: window length: 40 ms
                         window type: Hamming
                         overlap: window length/2 (50%)
                         num_fft_bins: window length (no zero padding)
    mask (optional): determines whether the output of RPCA is used directly as
                     separated sources (False), or it is used to create a mask
                     for separating the sources (True). Default is False                     
    maskgain (optional): a constant number which defines a kind of threshold
                         for computing the binary mask given low-rank and sparse
                         matrices (Mb(m,n) = 1 if |S(m,n)|>gain*|L(m,n)|).
                         Default is 1.
                    
    Output:
    y_f: M by N numpy array containing the foreground (sparse component), M 
         source images, each of length N time samples
    y_b: M by N numpy array containing the background (repeating/low-rak component),
         M source images, each of length N time samples    
         
    EXAMPLE:
     
    """
    
    # use the default range of repeating period and default do_STFT parameter values
    # if not specified
    if specparam is None:
       winlength=int(2**(np.ceil(np.log2(0.04*fs)))) # next power 2 of 40ms*fs
       specparam = [winlength,'Hamming',winlength/2,winlength]
   
   # do_STFT parameters
    winL,win,ovp,nfft=specparam 
    
    # compute the spectrograms of all channels
    M,N = np.shape(x)
    X=f_stft(np.array(x[0,:],ndmin=2),winL,win,ovp,fs,nfft,0)[0]
    for i in range(1,M):
         Sx = f_stft(np.mat(x[i,:]),winL,win,ovp,fs,nfft,0)[0]
         X=np.dstack([X,Sx])
         
    MagX=np.abs(X)  # magnitude spectrogram
    PhX=np.angle(X) # phase of the spectrogram
    if M==1: 
        X=X[:,:,np.newaxis]
        MagX=MagX[:,:,np.newaxis]
        PhX=PhX[:,:,np.newaxis]
    
    # decompose the magnitude spectrogram 
    L=1j*np.zeros((MagX.shape))
    S=1j*np.zeros((MagX.shape))
    for i in range(0,M):
        Li,Si = rpca(MagX[:,:,i])[0:2] # decompose the magnitude spectrogram
        L[:,:,i]=Li*np.exp(1j*PhX[:,:,i]) # append the phase of X to low-rank part
        S[:,:,i]=Si*np.exp(1j*PhX[:,:,i]) # append the phase of X to sparse part
    
    
    # compute the masks (using rpca) if specified    
    
    if mask==True:
        Mask=np.zeros((MagX.shape))     
        for i in range(0,M):
            Li=L[:,:,i]
            Si=S[:,:,i]
            
            MaskTemp=np.zeros((Li.shape))
            M_region=np.abs(Si)>maskgain*np.abs(Li) # find bins for which |Si|>gain*|Li|      
            MaskTemp[M_region]=1  # set the mask gain over the found region to one       
            Mask[:,:,i]=MaskTemp 
        
    # compute the separated sources
       
    yF=np.zeros((M,N)) # separated foreground
    yB=np.zeros((M,N)) # separated background   
    for i in range(0,M):    
       if mask == True:  
         XF=X[:,:,i]*Mask[:,:,i]
         XB=X[:,:,i]*(1-Mask[:,:,i])
       elif mask == False:
         XF=S[:,:,i]
         XB=L[:,:,i]
       
       yFi=f_istft(XF,winL,win,ovp,fs)[0]
       yF[i,:]=yFi[0:N]       
       
       yBi=f_istft(XB,winL,win,ovp,fs)[0]
       yB[i,:]=yBi[0:N] 
    
    
    return yF,yB


def rpca(M,delta=1e-7,maxit=100):
    """
    The function rpca implements the Robust Principle Component Analysis (RPCA) 
    method to decompose a matrix into its low rank and sparse components. 
    The augmented Lagrange multiplier (ALM) algorithm is used to solve the 
    optimization problem that outputs the decomposed parts. The optimization 
    problem can be stated as: min ||L||* + lambda ||S||1  s.t. L+S = M
    
    Inputs:
    M: Numpy n by m array containing the original matrix elements
    delta: stopping criterion
    maxit: maximum number of iterations
    
    Outputs:
    L: Numpy n by m array containing the elements of the low rank part
    S: Numpy n by m array containing the elements of the sparse part 
    
    """
    
    # compute the dimensions of the input matrix, M
    n,m=np.shape(M)    
    
    # compute the (rule of thumb) velues of the lagrange multiplyer and the 
    # svd-threshold 
    Lambda = 1/np.sqrt(np.max([n,m]))
    #mu = (n*m)/(4. * np.linalg.norm(M,ord=1)) 
        
    # initialize the low-rank matrix,L, sparse matrix,S, and the matrix of 
    # residuals, Y
    
    L=np.zeros((n,m))
    S=np.zeros((n,m))
    #Y=np.zeros((n,m))
    
    norm_two= np.linalg.svd(M, full_matrices=False,compute_uv=False)[0] 
    norm_inf=np.abs(M).max()/Lambda
    dual_norm=np.max([norm_two,norm_inf])
    Y=M/dual_norm
    
    # tunable parameters #########################
    mu=1.25/norm_two
    print(mu)
    mu_bar=mu*1e7
    rho=1.5
              
    # initialize the error value (loop condition)
    Etemp=1           
    #Etemp = np.linalg.norm(M - L - S,ord='fro') / np.linalg.norm(M,ord='fro')
    Error=np.array([],ndmin=2)
    
    converged=False
    It=0
    while converged==False: 
        It=It+1
        
        L = svd_thr(M - S + Y/mu,1/mu)
        S = shrink(M - L + Y/mu, Lambda/mu)
        Y = Y + mu*(M - L - S)
        
        mu=np.min([mu*rho,mu_bar])        ######################## 
        print(mu)
        
        Etemp = np.linalg.norm(M - L - S,ord='fro') / np.linalg.norm(M,ord='fro')
        Error=np.hstack([Error,np.array(Etemp,ndmin=2)])
        
        if Etemp<delta:
            converged=True
        if converged==False and It>=maxit:
            print('Maximum iteration reached')
            converged=True
    
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
    X: Numpy n by m array
    tau: shrinkage parameter 
    
    Output: 
    D_tau: Numpy n by m array containing elements of the matrix composed of 
           thresholded singular values
    """    
    
    U,sigma,V = np.linalg.svd(X, full_matrices=False) # truncated svd
    S_tau = shrink(sigma,tau)
    D_tau = np.dot(U, np.dot(np.diag(S_tau), V))
        
    return D_tau
    
    
def rd_svd(X,k):
    """
    The function rd_svd computes the matrix composed of a few largest components
    of the input matrix.
    
    Inputs:
    X: Numpy n by m array
    k: number of largest signular values to be selected
   
    Output:
    Xrd: reduced rank matrix composed of k largest components    
    """
    
    U,sigma,V = np.linalg.svd(X, full_matrices=False) # truncated svd
    S_rd=sigma[0:k]
    U_rd=U[:,0:k]
    V_rd=V[0:k,:]
    X_rd=np.dot(U_rd,np.dot(np.diag(S_rd),V_rd))
    
    return X_rd   
    
    
    
    
    
    
    
    
    
    