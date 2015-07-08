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
import matplotlib.pyplot 

def duet(*arg):
    """
    The "duet" function extracts N sources from a given stereo audio mixture 
    (N sources captured via 2 sensors)
    
    Inputs:
    x: a 2-row Numpy matrix containing samples of the two-channel mixture
    sparam: array containing spectrogram parameters including 
            L: window length (in # of samples)
          win: window type, string ('Rectangular', 'Hamming', 'Hanning', 'Blackman')
          ovp: number of overlapping samples between adjacent windows      
          nfft: min number of desired freq. samples in (-pi,pi]. MUST be >= L. 
               *NOTE* If this is not a power of 2, then it will automatically 
               zero-pad up to the next power of 2. IE if you put 257 here, 
               it will pad up to 512.
           fs: sampling rate of the signal
           ** sparam = np.array([L,win,ovp,nfft,fs])
           
    adparam: array containing ranges and number of bins for attenuation and delay 
           ** adparam = np.array([a_min,a_max,a_num,d_min,d_max,d_num]) 
    
    Pr: vector containing user defined information including a threshold value (in [0,1])
        for peak picking (thr), number of sources (N), and minimum distance between peaks
        ** Pr = np.array(thr,N,mindist)
    hist: (optional) string input, indicates if the histogram is to be plotted
          'y' (default): plot the histogram, 'n': don't plot    
          
    Output:
    y: an N-row Numpy matrix containing N time-domain estimates of sources
    ad_est: N by 2 Numpy matrix containing estimated attenuation and delay values
          corresponding to N sources   
    """
    # Extract the parameters from inputs
    if len(arg)<5: 
        x,sparam,adparam,Pr = arg[0:4]
        hist = 'y'    
    elif len(arg)==5:
        x,sparam,adparam,Pr,hist = arg[0:5]
    
    L=float(sparam[0]); win=sparam[1]; ovp=float(sparam[2]); nfft=int(sparam[3]);
    fs=int(sparam[4]);
    a_min=int(adparam[0]); a_max=int(adparam[1]); a_num=int(adparam[2]);
    d_min=int(adparam[3]); d_max=a_min=int(adparam[4]); d_num=int(adparam[5]);
    a_min,a_max,a_num,d_min,d_max,d_num = adparam[0:6]
    thr,N,mindist=Pr[0:3]
    
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
    
    # compute alpha and delta indices for the histogram
    #alpha_ind=np.round(1+(a_num-1)*(alpha_vec+a_max)/(2*a_max))-1
    #delta_ind=np.round(1+(d_num-1)*(delta_vec+a_max)/(2*d_max))-1
    
    # compute the histogram
    H=np.histogram2d(alpha_vec, delta_vec, bins=np.array([a_num,d_num]), range=np.array([[a_min,a_max],[d_min,d_max]]), normed=False, weights=tfw_vec)

#    H=np.zeros((a_num,d_num))
#    for i in range(0,len(alpha_vec)):
#        ai=alpha_ind[i]
#        di=delta_ind[i]
#        H[ai,di]=H[ai,di]+tfw_vec[i]
        
    # plot the 2D histogram
    hist=H[0]/H[0].max()   
    agrid=H[1]
    dgrid=H[2]
    
    AA=np.tile(agrid[1::],(d_num,1))
    DD=np.tile(dgrid[1::].T,(a_num,1)).T
    plt.pcolormesh(AA,DD,hist)
    plt.xlabel('alpha')
    plt.ylabel('delta')
    plt.show()

    
    
    return hist
    

