"""
This module implements the Kernel Additive Modeling (KAM) algorithm and its light 
version (KAML) for source separation. 

References:
[1] Liutkus, Antoine, et al. "Kernel additive models for source separation." 
    Signal Processing, IEEE Transactions on 62.16 (2014): 4298-4310.
[2] Liutkus, Antoine, Derry Fitzgerald, and Zafar Rafii. "Scalable audio 
    separation with light kernel additive modelling." IEEE International 
    Conference on Acoustics, Speech and Signal Processing (ICASSP). 2015.    
    
Required packages:
1. Numpy
2. Scipy
3. Matplotlib
4. Scikits Audiolab

Required modules:
1. f_stft
2. f_istft
"""

import numpy as np
import matplotlib.pyplot as plt
from scikits.audiolab import wavread, play
from f_stft import f_stft
from f_istft import f_istft
import time    

def kam(*arg):
    """
    The 'kam' function implements the kernel backfitting algorithm to extract 
    J audio sources from I channel mixtures.
    
    Inputs: 
    Inputfile: list of length 2. The first element is a string indicating the path of the .wav 
               file containing the I-channel audio mixture. The second element is the length
               of the mixture (in seconds) to be extracted and analyzed.
    SourceKernels: a list containg J sub-lists, each of which contains properties of 
                   one of source kernels. Kernel properties are:
                   -kernel type: (string) determines whether the kernel is one of the 
                                 pre-defined kernel types or a user-defined lambda function. 
                                 Choices are: 'cross','horizontal','vertical','periodic','userdef'
                   -kparams (for pre-defined kernels): a Numpy matrix containing the numerical 
                             values of the kernel parameters.
                   -knhood (for user-defined kernels): logical lambda function which defines 
                            receives the coordinates of two time-frequency bins and determines
                            whether they are neighbours (outputs TRUE if neighbour).
                   -kwfunc (optional): lambda function which receives the coordinates of two 
                            neighbouring time-frequency bins and computes the weight value at
                            the second bin given its distance from the first bin. The weight
                            values fall in the interval [0,1]. Default: all ones over the 
                            neighbourhood (binary kernel).
    Numit: (optional) number of iterations of the backfitting algorithm - default: 1
    
    Outputs:
    shat: a J by I by Ls Numpy matrix containing J time-domain source images on I channels        
    """
    
    if len(arg)==2:
        Inputfile,SourceKernels=arg[0:2]
        Numit=1
    elif len(arg)==3:
        Inputfile,SourceKernels,Numit=arg[0:3]
    
        
    # load the audio mixture from the input path
    Mixture=AudioSignal(Inputfile[0],Inputfile[1])
    x,tvec=np.array([Mixture.x,Mixture.time])  # time-domain channel mixtures
    X,Px,Fvec,Tvec=np.array([Mixture.X,Mixture.P,Mixture.Fvec,Mixture.Tvec])  # stft and PSD of the channel mixtures
    
    # Step (2): initialization
    # initialize the PSDs with average mixture PSD and the spatial covarince matricies
    # with identity matrices
    
    I=Mixture.numCh # number of channel mixtures
    J=len(SourceKernels) # number of sources
    LF=np.size(Fvec) # length of the frequency vector
    LT=np.size(Tvec) # length of the timeframe vector
    
    F_ind=np.arange(LF) # frequency bin indices
    T_ind=np.arange(LT) # time frame indices
    Fmesh,Tmesh=np.meshgrid(F_ind,T_ind) # grid of time-freq indices for the median filtering step
    TFcoords=np.mat(np.zeros((LF*LT,2),dtype=int)) # all time-freq index combinations
    TFcoords[:,0]=np.mat(np.asarray(Fmesh.T).reshape(-1)).T 
    TFcoords[:,1]=np.mat(np.asarray(Tmesh.T).reshape(-1)).T
    
   
    X=np.reshape(X.T,(LF*LT,I))  # reshape the STFT tensor into I vectors 
    MeanPSD=np.mean(Px,axis=2)/(I*J)
    MeanPSD=np.reshape(MeanPSD.T,(LF*LT,1)) # reshape the mean PSD matrix into a vector
        
    fj=np.zeros((LF*LT,I*I,J))
    for j in range(0,J):
       fj[:,:,j]=np.tile(MeanPSD,(1,I*I))  # initialize by mean PSD
    
           
    Rj=1j*np.zeros((1,I*I,J))
    for j in range(0,J):
        Rj[0,:,j]=np.reshape(np.eye(I),(1,I*I))
    Rj=np.tile(Rj,(LF*LT,1,1))   
    
    ### estimate sources from mixtures over Numit iterations
    
    S=1j*np.zeros((LF*LT,I,J))
    for n in range(0,Numit):
      
      # Step (3):
      # compute the inverse term: [sum_j' f_j' R_j']^-1
        SumFR=np.sum(fj*Rj,axis=2)
        SumFR.shape=(LF*LT,I,I)
        InvSumFR=np.reshape(np.linalg.inv(SumFR),(LF*LT,I*I))
        
        # compute sources, update PSDs and covariance matrices 
        for ns in range(0,J):
          FRinvsum=fj[:,:,ns]*Rj[:,:,ns]*InvSumFR
          Stemp=1j*np.zeros((LF*LT,I))
          for nch in range(0,I):
              FRtemp=FRinvsum[:,nch*I:nch*I+2]
              Stemp[:,nch]=np.sum(FRtemp*X,axis=1)
          S[:,:,ns]=Stemp
          
          # Step (4-a):
          Cj=np.repeat(Stemp,I,axis=1)*np.tile(np.conj(Stemp),(1,I))
          
          # Step (4-b):
          Cj_reshape=np.reshape(Cj,(LF*LT,I,I))     
          Cj_trace=np.mat(np.matrix.trace(Cj_reshape.T)).T
          MeanCj=Cj/np.tile(Cj_trace,(1,I*I))
          MeanCj_reshape=np.reshape(np.array(MeanCj),(LF,LT,I*I),order='F') 
          Rj[:,:,ns]=np.tile(np.sum(MeanCj_reshape,axis=1),(LT,1))
          
          # Step (4-c):
          # Note: the summation over 't' at step 4-c in the 2014 paper is a typo!
          #       the correct formulation of zj is: 
          #       zj=(1/I)*tr(inv(Rj(w)Cj(w,t)
          Rj_reshape=np.reshape(Rj[:,:,ns],(LF*LT,I,I))
          InvRj=np.reshape(np.linalg.inv(Rj_reshape),(LF*LT,I*I))
          InvRjCj=np.reshape(InvRj*Cj,(LF*LT,I,I))
          zj=np.real(np.matrix.trace(InvRjCj.T)/I) 
          zj=np.mat(zj) 
          
          # Step (4-d):
          # Extract the source kernel type and properties:
          SKj=SourceKernels[ns]
          if len(SKj)<2:
              raise Exception('The information required for generating source kernels is insufficient.'\
                               ' Each sub-list in SourceKernels must contain at least two elements.') 
          elif len(SKj)==2:
              Kj=Kernel(SKj[0],SKj[1])
          elif len(SKj)==3:
              Kj=Kernel(SKj[0],SKj[1],SKj[2])
           
          start_time = time.clock()
                           
          for ft in range(0,LF*LT):
              simTemp=Kj.sim(TFcoords[ft],TFcoords)
              NhoodTemp=np.nonzero(simTemp)
              zjNhood=np.multiply(zj[NhoodTemp],simTemp[NhoodTemp])
              fj[ft,:,ns]=np.median(np.array(zjNhood))
              
          print time.clock() - start_time, "seconds"    
    
    
    # Reshape the spectrograms
    Shat=1j*np.zeros((LF,LT,I,J)) # estimated source STFTs
    for ns in range(0,J):
        for nch in range(0,I):
            Shat[:,:,nch,ns]=np.reshape(S[:,nch,ns],(LT,LF)).T
            
    # Compute the inverse STFT of the estimated sources
    shat=np.zeros((x.shape[0],I,J))
    sigTemp=AudioSignal()
    sigTemp.numCh=I
    for ns in range(0,J):
        sigTemp.X=Shat[:,:,:,ns]
        shat[:,:,ns]=sigTemp.iSTFT()[0][0:x.shape[0]]
    
    return shat,fj,Rj          
          
 

class AudioSignal:   
    """
    The class Signal defines the properties of the audio signal object and performs
    basic operations such as Wav loading and computing the STFT/iSTFT.
    
    Read/write signal properties:
    - s: signal
    - sigLen: signal length (in number of samples)
    
    Read/write stft properties:
    - windowtype (e.g. 'Rectangular', 'Hamming', 'Hanning', 'Blackman')
    - windowlength (ms)
    - nfft (number of samples)
    - overlapRatio (in [0,1])
    - S: stft of the data
        
    Read-only properties:
    - fs: sampling frequency
    - enc: encoding of the audio file
    - numCh: number of channels
  
    EXAMPLES:
    -create a new signal object:     sig=Signal('sample_audio_file.wav')  
    -compute the spectrogram of the new signal object:   sigSpec,sigPow,F,T=sig.STFT()
    -compute the inverse stft of a spectrogram:          sigrec,tvec=sig.iSTFT()
  
    """
    def __init__(self,*arg):
                
        # signal properties
        self.x=np.mat([])
        self.fs = 44100
        self.time=np.mat([])
        self.enc='pcm16'
        self.sigLen=0
        self.numCh=0
              
        # STFT properties
        self.X=np.mat([]) # complex spectrogram
        self.P=np.mat([]) # power spectrogram
        self.Fvec=np.mat([]) # freq. vector
        self.Tvec=np.mat([]) # time vector
        self.windowtype = 'Hamming'
        self.windowlength = int(0.06*self.fs)
        self.nfft = self.windowlength
        self.overlapRatio = 0.5
        self.overlapSamp=int(np.ceil(self.overlapRatio*self.windowlength))
        self.makeplot = 1
        self.fmaxplot=self.fs/2
        
        if len(arg)==0:
            return
        elif len(arg)==1:
            self.loadaudiofile(arg[0])            
        elif len(arg)==2:
            if type(arg[0])==str:
               self.loadaudiofile(arg[0],arg[1])
            else:
               self.loadaudiosig(arg[0],arg[1])
             
        self.STFT()
                
   
    
    def loadaudiofile(self,*arg):
        """
        loads the audio signal from a .wav file
        inputs: 
        file_name is a string argument indicating the name of the .wav file
        siglen (in seconds): optional input indicating the length of the signal to be extracted
        """
        
        if len(arg)==1:
           file_name=arg[0]
           siglen='full length'
        elif len(arg)==2:
           file_name,siglen=arg[0:2]
        
        x,self.fs,self.enc = wavread(file_name)
        self.numCh=np.shape(x)[1]
        if siglen=='full length':
           self.x=np.mat(x) # make sure the signal is of matrix format
           self.sigLen= np.shape(x)[0]           
        else:
           self.sigLen=int(np.floor(siglen*self.fs))
           self.x=np.mat(x[0:self.sigLen,:])
           
        self.time=np.mat((1./self.fs)*np.arange(self.sigLen))
        
        
        
    def loadaudiosig(self,audiosig,fs):
        """
        loads the audio signal in numpy matrix format along with the sampling frequency
        """
        self.x=np.mat(audiosig) # each column contains one channel mixture
        self.fs = fs
        self.sigLen,self.numCh=np.shape(self.x)
        self.time=np.mat((1./self.fs)*np.arange(self.sigLen))
        
        
    def STFT(self):
        """
        computes the STFT of the audio signal and returns:
        self.X: complex stft
        self.P: power spectrogram
        self.Fvec: frequency vector
        self.Tvec: vector of time frames
        """
        
        for i in range(0,self.numCh):   
            Xtemp,Ptemp,Ftemp,Ttemp = f_stft(self.x[:,i].T,self.windowlength,self.windowtype,
                       self.overlapSamp,self.nfft,self.fs,0)
                       
            if np.size(self.X) ==0:
                self.X=Xtemp
                self.P=Ptemp
                self.Fvec=Ftemp
                self.Tvec=Ttemp
            else:
                self.X=np.dstack([self.X,Xtemp]) 
                self.P=np.dstack([self.P,Ptemp])
        
        if self.makeplot==1:
            plt.figure()
            f_stft(self.x[:,i].T,self.windowlength,self.windowtype,
                       np.ceil(self.overlapRatio*self.windowlength),self.nfft,self.fs,self.makeplot,self.fmaxplot)
            plt.show()
            
        return self.X,self.P,self.Fvec,self.Tvec
            
    
    def iSTFT(self):
         """
         computes the inverse STFT and returns:
         self.x: time-domain signal       
         self.time: time vector 
         """
         
         if np.size(self.X)==0: 
             print("Empty spectrogrm matrix!")
             self.x=np.mat([])
             self.time=np.mat([])
         else:         
             self.x=np.mat([])
             for i in range(0,self.numCh):
                 x_temp,t_temp=f_istft(self.X[:,:,i],self.windowlength,self.windowtype,self.overlapSamp,
                                self.nfft,self.fs)
             
                 if np.size(self.x)==0:
                     self.x=np.mat(x_temp).T
                     self.time=np.mat(t_temp).T
                 else:
                     self.x=np.hstack([self.x,np.mat(x_temp).T])
                     
         return self.x,self.time


class Kernel:
    """
    The class Kernel defines the properties of the time-freq proximity kernel. The weight values of 
    the proximity kernel over time-frequecy bins that are considered as neighbours are given
    by a pre-defined or a user-defined function. The value of the proximity kernel is zero over
    time-frequency bins outside the neighbourhood.
    
    Properties:
    
    -kType: (string) determines whether the kernel is one of the pre-defined kernel types 
             or a user-defined lambda function. 
             Predefined choices are: 'cross','horizontal','vertical','periodic'
             To define a new kernel type, kType should be set to: 'userdef'
             
    -kParamVal: a Numpy matrix containing the numerical values of the kernel parameters. If any
             of the pre-defined kernel type is selected, the parameter values should be provided 
             through kParamVal. Parameters corresponding to the pre-defined kernels are:
             Cross: (neighbourhood width along the freq. axis in # of freq. bins, neighbour width
                     along the time axis in # of time frames)
             Vertical: (neighbourhood width along the freq. axis in # of freq. bins)
             Horizontal: (neighbourhood width along the time axis in # of time frames)
             Periodic: (period in # of time frames,neighbourhood width along the time axis 
                        in # of frames)  
             Note: neighbourhood width is measured in only one direction, e.g. only to the
                   right of a time-freq bin in the case of a horizontal kernel, so the whole
                   length of the neighbourhood would be twice the specified width.
             
    -kNhood: logical lambda funcion which receives the coordinates of two time-frequency
             bins and determines whether they are neighbours (outputs TRUE if neighbour).
             
    -kWfunc: lambda function which receives the coordinates of two time-frequency bins that are
             considered neighbours by kNhood and computes the weight value at the second bin given 
             its distance from the first bin. The weight values fall in the interval [0,1] with 
             1 indicating zero-distance or equivalently perfect similarity. 
             Default: all ones over the neighbourhood (binary kernel)
    
    EXAMPLE: 
    
    FF,TT=np.meshgrid(np.arange(5),np.arange(7))
    TFcoords1=np.mat('2,3')
    TFcoords2=np.mat(np.zeros((35,2)))
    TFcoords2[:,0]=np.mat(np.asarray(FF.T).reshape(-1)).T
    TFcoords2[:,1]=np.mat(np.asarray(TT.T).reshape(-1)).T

    W=lambda TFcoords1,TFcoords2: np.exp(-(TFcoords1-TFcoords2)*(TFcoords1-TFcoords2).T)
    k_cross=Kernel('cross',np.mat([3,2]),W)
    simVal_cross=np.reshape(k_cross.sim(TFcoords1,TFcoords2),(5,7))
                      
    """
    
    def __init__(self,*arg):
                
        # kernel properties
        self.kType='' # default: no pre-defined kernel selected
        self.kParamVal=np.mat([])      
        self.kNhood=lambda TFcoords1,TFcoords2: (TFcoords1==TFcoords2).all() # default: neighnourhood includes only the centeral bin      
        self.kWfunc=lambda TFcoords1,TFcoords2: float(self.kNhood(TFcoords1,TFcoords2)) # default: binary kernel
        
 
        if len(arg)==0:
           return
        elif len(arg)==1:
            self.genkernel(arg[0])
        elif (len(arg)==2):
           self.genkernel(arg[0],arg[1]) 
        elif len(arg)==3: 
           self.genkernel(arg[0],arg[1],arg[2]) 

    
    def genkernel(self,*arg):
        """
        generates the kernel object given the user-defined properties
        
        Inputs:
        Type: (string) determines whether the kernel is one of the pre-defined kernel types 
             or a user-defined lambda function. 
             Predefined choices are: 'cross','horizontal','vertical','periodic'
             To define a new kernel type, kType should be set to: 'userdef'
             
        ParamVal: a Numpy matrix containing the numerical values of the kernel parameters. If any
             of the pre-defined kernel type is selected, the parameter values should be provided 
             through kParamVal. Parameters corresponding to the pre-defined kernels are:
             Cross: (neighbourhood width along the freq. axis in # of freq. bins, neighbour width
                     along the time axis in # of time frames)
             Vertical: (neighbourhood width along the freq. axis in # of freq. bins)
             Horizontal: (neighbourhood width along the time axis in # of time frames)
             Periodic: (period in # of time frames,neighbourhood width along the time axis 
                        in # of frames)  
             
        Nhood: logical lambda funcion which receives the coordinates of two time-frequency
             bins and determines whether they are neighbours (outputs TRUE if neighbour).
             
        Wfunc: lambda function which receives the coordinates of two time-frequency bins that are
             considered neighbours by kNhood and computes the weight value at the second bin given 
             its distance from the first bin. The weight values fall in the interval [0,1] with 
             1 indicating zero-distance or equivalently perfect similarity. 
             Default: all ones over the neighbourhood (binary kernel)
        """
   
        if len(arg)==0:
            Type=self.kType
            ParamVal=self.kParamVal
            Nhood=self.kNhood
            Wfunc=self.kWfunc
        elif len(arg)==1:
            Type=arg[0]
            ParamVal=self.kParamVal
            Nhood=self.kNhood
            Wfunc=self.kWfunc
            if Type!='userdef':
               print('Warning: kernel parameter values are not specified.')
            elif Type=='userdef':
               print('Warning: kernel neighbourhood is not defined.')
        elif len(arg)==2:
            if (arg[0] in ['cross','horizontal','vertical','periodic']):
                Type=arg[0]
                ParamVal=arg[1]
            elif arg[0]=='userdef':    
                Type=arg[0]
                Nhood=arg[1]
        elif len(arg)==3:
            if (arg[0] in ['cross','horizontal','vertical','periodic']):
                Type,ParamVal,Wfunc=arg[0:3]
            elif arg[0]=='userdef':
                Type,Nhood,Wfunc=arg[0:3]
                
        self.kType=Type
        self.kParamVal=ParamVal
        
        if len(arg)>1:
            if Type=='cross':
               Df=ParamVal[0,0]
               Dt=ParamVal[0,1]                           
               self.kNhood=lambda TFcoords1,TFcoords2: np.logical_or(np.logical_and((np.tile(TFcoords1[:,0],(1,TFcoords2.shape[0]))==np.tile(TFcoords2[:,0].T,(TFcoords1.shape[0],1))),\
                            (np.abs(np.tile(TFcoords1[:,1],(1,TFcoords2.shape[0]))-np.tile(TFcoords2[:,1].T,(TFcoords1.shape[0],1)))<Dt)),\
                            np.logical_and((np.tile(TFcoords1[:,1],(1,TFcoords2.shape[0]))==np.tile(TFcoords2[:,1].T,(TFcoords1.shape[0],1))),\
                            (np.abs(np.tile(TFcoords1[:,0],(1,TFcoords2.shape[0]))-np.tile(TFcoords2[:,0].T,(TFcoords1.shape[0],1)))<Df)))
                          
            elif Type=='vertical':
               Df=ParamVal[0,0]               
               self.kNhood=lambda TFcoords1,TFcoords2: np.logical_and((np.tile(TFcoords1[:,1],(1,TFcoords2.shape[0]))==np.tile(TFcoords2[:,1].T,(TFcoords1.shape[0],1))),\
                            (np.abs(np.tile(TFcoords1[:,0],(1,TFcoords2.shape[0]))-np.tile(TFcoords2[:,0].T,(TFcoords1.shape[0],1)))<Df))
                              
            elif Type=='horizontal':
               Dt=ParamVal[0,0]               
               self.kNhood=lambda TFcoords1,TFcoords2: np.logical_and((np.tile(TFcoords1[:,0],(1,TFcoords2.shape[0]))==np.tile(TFcoords2[:,0].T,(TFcoords1.shape[0],1))),\
                            (np.abs(np.tile(TFcoords1[:,1],(1,TFcoords2.shape[0]))-np.tile(TFcoords2[:,1].T,(TFcoords1.shape[0],1)))<Dt))
                                     
            elif Type=='periodic':
               P=ParamVal[0,0]
               Dt=ParamVal[0,1]
               self.kNhood=lambda TFcoords1,TFcoords2: np.logical_and(np.logical_and((np.tile(TFcoords1[:,0],(1,TFcoords2.shape[0]))==np.tile(TFcoords2[:,0].T,(TFcoords1.shape[0],1))),\
                            (np.abs(np.tile(TFcoords1[:,1],(1,TFcoords2.shape[0]))-np.tile(TFcoords2[:,1].T,(TFcoords1.shape[0],1)))<Dt)),\
                            (np.mod(np.tile(TFcoords1[:,1],(1,TFcoords2.shape[0]))-np.tile(TFcoords2[:,1].T,(TFcoords1.shape[0],1)),P)==0))         
                            
            elif Type=='userdef':
               self.kNhood=Nhood
               
  
        if len(arg)==2:
             self.kWfunc=lambda TFcoords1,TFcoords2: self.kNhood(TFcoords1,TFcoords2).astype(np.float)
        elif len(arg)==3:
            self.kWfunc=Wfunc
            
            
            
        
    def sim(self,TFcoords1,TFcoords2):
         """
         Measures the similarity between a series of new time-freq points and the kernel central point.
         
         Inputs:
         TFcoords1: N1 by 2 Numpy matrix containing coordinates of N1 time-frequency bins.
                    Each row contains the coordinates of a single bin.
         TFcoords2: N2 by 2 Numpy matrix containing coordinates of N2 time-frequency bins.
         
         Output:
         simVal: N1 by N2 Numby matrix of similarity values. Similarity values fall in the interval [0,1].
                 The value of the (i,j) element in simVal determines the amountof similarity (or closeness) 
                 between the i-th time-frequency bin in TFcoords1 and j-th time-frequency bin in TFcoords2. 
         """         
         
         self.genkernel() # update the kernel with possibly altered properties
         
         N1=np.shape(TFcoords1)[0]
         N2=np.shape(TFcoords2)[0]
         
         simVal=np.zeros((N1,N2))
         
         Nhood_vec=self.kNhood(TFcoords1,TFcoords2)
         Wfunc_vec=self.kWfunc(TFcoords1,TFcoords2)
         simVal=np.multiply(Nhood_vec,Wfunc_vec).astype(np.float)
  
#         for i in range(0,N1):
#                Nhood_i=self.kNhood(TFcoords1[i,:],TFcoords2)
#                Wfunc_i=self.kWfunc(TFcoords1[i,:],TFcoords2)
#                simVal[i,:]=np.multiply(Nhood_i,Wfunc_i).astype(np.float)
                
   
         return simVal
             
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
   
        
        
        
        
        
        
        
        
        
   