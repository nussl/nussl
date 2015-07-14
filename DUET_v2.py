"""
This module implements the Degenerate Unmixing Estimation Technique (DUET) algorithm.
The DUET algorithm was originally proposed by S.Rickard and F.Dietrich for DOA estimation
and further developed for BSS and demixing by A.Jourjine, S.Rickard, and O. Yilmaz.

References:
[1] Rickard, Scott. "The DUET blind source separation algorithm." Blind Speech Separation. 
    Springer Netherlands, 2007. 217-241.
[2] Yilmaz, Ozgur, and Scott Rickard. "Blind separation of speech mixtures via 
    time-frequency masking." Signal Processing, IEEE transactions on 52.7 
    (2004): 1830-1847.

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
from mpl_toolkits.mplot3d import axes3d
from scipy import signal

def duet_v2(*arg):
    """
    The 'duet_v2' function extracts N sources from a given stereo audio mixture 
    (N sources captured via 2 sensors). It differs from the 'duet' function in that
    instead of using an automated peak-picking process, it asks the user to click
    on the peaks of the 2D histogram and therefore provide the algorithm with exact
    peak coordinates.
    
    Inputs:
    x: a 2-row Numpy matrix containing samples of the two-channel mixture
    sparam: structure array containing spectrogram parameters including 
            L: window length (in # of samples)
          win: window type, string ('Rectangular', 'Hamming', 'Hanning', 'Blackman')
          ovp: number of overlapping samples between adjacent windows      
          nfft: min number of desired freq. samples in (-pi,pi]. MUST be >= L. 
               *NOTE* If this is not a power of 2, then it will automatically 
               zero-pad up to the next power of 2. IE if you put 257 here, 
               it will pad up to 512.
           fs: sampling rate of the signal
           ** sparam = np.array([(L,win,ovp,nfft,fs)]
           dtype=[('winlen',int),('wintype','|S10'),('overlap',int),('numfreq',int),('sampfreq',int)])
           
    adparam: structure array containing ranges and number of bins for attenuation and delay 
           ** adparam = np.array([(a_min,a_max,a_num,d_min,d_max,d_num)],
           dtype=[('amin',float),('amax',float),('anum',float),('dmin',float)
           ,('dmax',float),('dnum',int)])
    
    plothist: (optional) string input, indicates if the 3D histogram is to be plotted
          'y' (default): plot the histogram, 'n': don't plot    
          
    Output:
    xhat: an N-row Numpy matrix containing N time-domain estimates of sources
    ad_est: N by 2 Numpy matrix containing estimated attenuation and delay values
          corresponding to N sources   
    """
    # Extract the parameters from inputs
    if len(arg)<5: 
        x,sparam,adparam = arg[0:3]
        plothist = 'y'    
    elif len(arg)==5:
        x,sparam,adparam,plothist = arg[0:4]
    
    sparam = sparam.view(np.recarray)
    adparam = adparam.view(np.recarray)
    L=sparam.winlen; win=sparam.wintype; ovp=sparam.overlap; nfft=sparam.numfreq; fs=sparam.sampfreq;
    a_min=adparam.amin; a_max=adparam.amax; a_num=adparam.anum;
    d_min=adparam.dmin; d_max=adparam.dmax; d_num=adparam.dnum;
        
    # Compute the STFT of the two channel mixtures
    X1,P1,F,T = f_stft(x[0,:],L,win,ovp,nfft,fs,0)
    X2,P2,F,T = f_stft(x[1,:],L,win,ovp,nfft,fs,0)
    # remove dc component to avoid dividing by zero freq. in the delay estimation
    X1=X1[1::,:]; X2=X2[1::,:]; 
    Lf=len(F); Lt=len(T);
    
    # Compute the freq. matrix for later usein phase calculations
    wmat=np.array(np.tile(np.mat(F[1::]).T,(1,Lt)))*(2*np.pi/fs)
    
    # Calculate the symmetric attenuation (alpha) and delay (delta) for each 
    # time-freq. point
    R21 = (X2+1e-16)/(X1+1e-16)
    atn = np.abs(R21) # relative attenuation between the two channels
    alpha = atn - 1/atn # symmetric attenuation
    delta = -np.imag(np.log(R21))/(2*np.pi*wmat) # relative delay    
    
    # calculate the weighted histogram
    p=1; q=0;
    tfw = (np.abs(X1)*np.abs(X2))**p * (np.abs(wmat))**q #time-freq weights
    
    # only consider time-freq. points yielding estimates in bounds
    a_premask=np.logical_and(a_min<alpha,alpha<a_max)
    d_premask=np.logical_and(d_min<delta,delta<d_max)
    ad_premask=np.logical_and(a_premask,d_premask)
    
    ad_nzind=np.nonzero(ad_premask)
    alpha_vec=alpha[ad_nzind]
    delta_vec=delta[ad_nzind]
    tfw_vec=tfw[ad_nzind]
    
    # compute the histogram
    H=np.histogram2d(alpha_vec, delta_vec, bins=np.array([a_num[0],d_num[0]]), 
                     range=np.array([[a_min,a_max],[d_min,d_max]]), normed=False, weights=tfw_vec)

    # plot the 2D histogram
    hist=H[0]/H[0].max()   
    agrid=H[1]
    dgrid=H[2]
    
    # smooth the histogram - local average 3-by-3 neightboring bins 
    hist=twoDsmooth(hist,3)
    
    # normalize the histogram
    hist=hist/hist.max()
    
    AA=np.tile(agrid[1::],(d_num,1)).T
    DD=np.tile(dgrid[1::].T,(a_num,1))
    if plothist=='y':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(AA, DD, hist, rstride=2, cstride=2)
        plt.xlabel(r'$\alpha$',fontsize=16)
        plt.ylabel(r'$\delta$',fontsize=16)
        plt.title(r'$\alpha-\delta$ Histogram')
        plt.axis('tight')
        ax.view_init(30, 30)
        plt.draw()
    
    # plot the histogram in 2D space and ask the user to click
    # on the histogram peaks
    print("Click on histogram peaks. Press enter to exit.")
    fig = plt.figure()
    plt.pcolormesh(AA,DD,hist)
    plt.xlabel(r'$\alpha$',fontsize=16)
    plt.ylabel(r'$\delta$',fontsize=16)
    plt.title(r'$\alpha-\delta$ Histogram')
    plt.axis('tight')
    plt.show()
    
    
    plocation = np.array(plt.ginput(0,0)).T
    N=np.shape(plocation)[1]
        
    # use the peak coordinates entered by the user for generating the masks
    alphapeak=plocation[0,:]
    deltapeak=plocation[1,:]
    
    ad_est=np.vstack([alphapeak,deltapeak]).T
   
    #convert alpha to a 
    atnpeak=(alphapeak+np.sqrt(alphapeak**2+4))/2
    
    # compute masks for separation
    bestsofar=np.inf*np.ones((Lf-1,Lt)) 
    bestind=np.zeros((Lf-1,Lt),int)
    for i in range(0,N):
        score=np.abs(atnpeak[i]*np.exp(-1j*wmat*deltapeak[i])*X1-X2)**2/(1+atnpeak[i]**2)
        mask=(score<bestsofar)
        bestind[mask]=i
        bestsofar[mask]=score[mask]
        
    # demix with ML alignment and convert to time domain    
    Lx=np.shape(x)[1]    
    xhat=np.zeros((N,Lx))
    for i in range(0,N):
        mask=(bestind==i)
        Xm=np.vstack([np.zeros((1,Lt)),(X1+atnpeak[i]*np.exp(1j*wmat*deltapeak[i])*X2)
        /(1+atnpeak[i]**2)*mask])
        xi=f_istft(Xm,L,win,ovp,nfft,fs)
        
        xhat[i,:]=np.array(xi)[0,0:Lx]
        # add back to the separated signal a portion of the mixture to eliminate
        # most of the masking artifacts
        #xhat=xhat+0.05*x[0,:]
        
    return xhat,ad_est
    
 
def twoDsmooth(Mat,Kernel):
    """
    The 'twoDsmooth' function receivees a matrix and a kernel type and performes
    two-dimensional convolution in order to smooth the values of matrix elements.
    (similar to low-pass filtering)
    
    Inputs:
    Mat: a 2D Numpy matrix to be smoothed 
    Kernel: a 2D Numpy matrix containing kernel values
           Note: if Kernel is of size 1 by 1 (scalar), a Kernel by Kernel matrix
           of 1/Kernel**2 will be used as teh matrix averaging kernel
    Output:
    SMat: a 2D Numpy matrix containing a smoothed version of Mat (same size as Mat)                 
    """
    
    # check the dimensions of the Kernel matrix and set the values of the averaging
    # matrix, Kmat
    if np.prod(np.shape(Kernel))==1:
       Kmat= np.ones((Kernel,Kernel))/Kernel**2
    else:
        Kmat = Kernel
       
    # make Kmat have odd dimensions
    krow,kcol = np.shape(Kmat)
    if np.mod(krow,2)==0:
        Kmat=signal.convolve2d(Kmat,np.ones((2,1)))/2
        krow=krow+1
        
    if  np.mod(kcol,2)==0:
         Kmat=signal.convolve2d(Kmat,np.ones((1,2)))/2
         kcol=kcol+1
         
    # adjust the matrix dimension for convolution
    matrow,matcol = np.shape(Mat)
    copyrow=int(np.floor(krow/2)) # number of rows to copy on top and bottom
    copycol=int(np.floor(kcol/2)) # number of columns to copy on either side
    
    # form the augmented matrix (rows and columns added to top, botoom, and sides)
    Mat=np.mat(Mat) # make sure Mat is a Numpy matrix
    augMat=np.vstack([np.hstack([Mat[0,0]*np.ones((copyrow,copycol)),np.ones((copyrow,1))*Mat[0,:],Mat[0,-1]*np.ones((copyrow,copycol))])
    ,np.hstack([Mat[:,0]*np.ones((1,copycol)),Mat,Mat[:,-1]*np.ones((1,copycol))])
    ,np.hstack([Mat[-1,1]*np.ones((copyrow,copycol)),np.ones((copyrow,1))*Mat[-1,:],Mat[-1,-1]*np.ones((copyrow,copycol))])])     
    
    # perform two-dimensional convolution between the input matrix and the kernel
    SMAT= signal.convolve2d(augMat,Kmat[::-1,::-1],mode='valid')
    
    return SMAT
    
    
    

    


    
    
