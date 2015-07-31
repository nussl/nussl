"""
This module implements the Kernel Additive Modeling (KAM) algorithm for source
separation. 

Reference:
[1] Liutkus, Antoine, et al. "Kernel additive models for source separation." 
    Signal Processing, IEEE Transactions on 62.16 (2014): 4298-4310.
    
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

def kam(*arg):
    """
    The 'kam' function implements the kernel backfitting algorithm to extract 
    J audio sources from I channel mixtures.
    
    Inputs: 
    Inputfile: (strig) path of the audio file containing the I-channel mixture
    KNhoods: Numpy array of length J - each element is a lambda function defining 
             the proximity kernel for one audio source
    KWeights: (optional) Numpy array of length J - each element gives the similarity measure 
              over the corresponding neighbourhood provided by KNhoods. A similarity
              measure can be either a lambda function (varying weights over neighbouring
              points) or equal to 1 (equal weight over the entire neighbourhood).
              Default: all weights are equal to 1 (binary kernels).
    Numit: (optional) number of iterations of the backfitting algorithm - default: 1
    
    Outputs:
    shat: a J-row Numpy matrix containing J time-domain estimates of sources        
    """
    
    if len(arg)==2:
        Inputfile,KNhoods=arg[0:2]        
        KWeights=np.ones((1,len(KNhoods)))
        Numit=1
    elif len(arg)==3:
        Inputfile,KNhoods,KWeights=arg[0:3]
        J=len(KNhoods)
        Numit=1
    elif len(arg)==4:
        Inputfile,KNhoods,KWeights,Numit=arg[0:4]
        
        
    # load the audio mixture from the input path
    Mixture=AudioSignal(Inputfile) 
    x,tvec=np.array([Mixture.x,Mixture.time])  # time-domain channel mixtures
    X,Px,Fvec,Tvec=np.array([Mixture.X,Mixture.P,Mixture.Fvec,Mixture.Tvec])  # stft and PSD of the channel mixtures
    
    # initialization
    # initialize the PSDs with average mixture PSD and the spatial covarince matricies
    # with identity matrices
    
    I=Mixture.numCh # number of channel mixtures
    J=np.size(KNhoods) # number of sources
    LF=np.size(Fvec) # length of the frequency vector
    LT=np.size(Tvec) # length of the timeframe vector
   
    fj=np.zeros((LF,LT,J))
    fj[:,:,0]=np.mean(Px,axis=2)/J
    for j in range(1,J):
        fj[:,:,j]=fj[:,:,0]
        
    Rj=np.zeros((1,I,I,J))
    for j in range(0,J):
        Rj[0,:,:,j]=np.eye(I)
    Rj=np.tile(Rj,(LF,1,1,1))    
    
    ### estimate sources from mixtures over Numit iterations
    
    S=np.zeros((LF,LT,J))
    for i in range(0,Numit):
      print(i)  
    
    
    
    #return x,X,Px,Fvec,Tvec,tvec


class AudioSignal:   
    """
    The class Signal defines the properties of the audio signal object and performs
    basic operations such as Wav loading and computing the STFT/iSTFT.
    
    Read/write signal properties:
    - s: signal
    - sigLen: signal length
    
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
            self.STFT()
        elif len(arg)==2:
             self.loadaudiosig(arg[0],arg[1])
             self.STFT()
                
   
    
    def loadaudiofile(self,file_name):
        """
        loads the audio signal from a .wav file
        input: file_name is a string argument indicating the name of the .wav file
        """
        self.x,self.fs,self.enc = wavread(file_name)
        self.x=np.mat(self.x) # make sure the signal is of matrix format
        self.sigLen,self.numCh = np.shape(self.x)
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
             
    -kNhood: logical lambda funcion which receives the coordinates of two time-frequency
             bins and determines whether they are neighbours (outputs TRUE if neighbour).
             
    -kWfunc: lambda function which receives the coordinates of two time-frequency bins that are
             considered neighbours by kNhood and computes the weight value at the second bin given 
             its distance from the first bin. The weight values fall in the interval [0,1] with 
             1 indicating zero-distance or equivalently perfect similarity. 
             Default: all ones over the neighbourhood (binary kernel)
    

    
                  
    """
    
    def __init__(self,*arg):
                
        # kernel properties
        self.kType='' # default: no pre-defined kernel selected
        self.kParamVal=np.mat([])      
        self.kNhood=lambda TFcoords1,TFcoords2: (TFcoords1==TFcoords2).all() # default: neighnourhood includes only the centeral bin      
        self.kWfunc=lambda TFcoords1,TFcoords2: float(self.kNhood(TFcoords1,TFcoords2)) # default: binary kernel
        
 
        if len(arg)==0:
           return
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
        elif len(arg)==2:
            Wfunc=self.kWfunc
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
                
                      
        if Type=='cross':
           Df=ParamVal[0,0]
           Dt=ParamVal[0,1]
           self.kNhood=lambda TFcoords1,TFcoords2: (TFcoords1[0,0]==TFcoords2[0,0] and np.abs(TFcoords1[0,1]-TFcoords2[0,1])<Df)\
                         or (TFcoords1[0,1]==TFcoords2[0,1] and np.abs(TFcoords1[0,0]-TFcoords2[0,0])<Dt) 
           self.kWfunc=Wfunc              
        elif Type=='vertical':
           Df=ParamVal[0,0]
           self.kNhood=lambda TFcoords1,TFcoords2: (TFcoords2[0,1]==TFcoords1[0,1] and np.abs(TFcoords2[0,0]-TFcoords1[0,0])<Df) 
           self.kWfunc=Wfunc  
        elif Type=='horizontal':
           Dt=ParamVal[0,0]
           self.kNhood=lambda TFcoords1,TFcoords2: (TFcoords2[0,0]==TFcoords1[0,0] and np.abs(TFcoords2[0,1]-TFcoords1[0,1])<Dt) 
           self.kWfunc=Wfunc  
        elif Type=='periodic':
           P=ParamVal[0,0]
           Dt=ParamVal[0,1]
           self.kNhood=lambda TFcoords1,TFcoords2: (TFcoords2[0,0]==TFcoords1[0,0] and np.abs(TFcoords2[0,1]-TFcoords1[0,1])<Dt\
                         and np.mod(TFcoords2[0,1]-TFcoords1[0,1],P)==0 ) 
           self.kWfunc=Wfunc  
        elif Type=='userdef':
           self.kNhood=Nhood
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
         for i in range(0,N1):
             for j in range(0,N2):
                Nhood_ij=self.kNhood(TFcoords1[i,:],TFcoords2[j,:])
                Wfunc_ij=self.kWfunc(TFcoords1[i,:],TFcoords2[j,:])
                simVal[i,j]=float(Nhood_ij*Wfunc_ij)
    
         return simVal
             
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#class Kernelmat:   
#    """
#    The class Kernelmat defines the properties of the proximity kernel, a matrix with values
#    equal to one for elements that are considered neightbours and zero values elsewhere.
#    The proximity kernel accounts for features like periodicity, continuity, smoothness, stability over time or frequency,
#    self-similarity, etc.
#    
#    Properties:
#    -kdim: 1 by 2 Numpy matrix containing kernel dimensions (number of rows and columns)
#    -kcenter: 1 by 2 Numpy matrix containing row and column numbers of the kernel center 
#             (distance is measured with respect to this central element)            
#    -knhood: kernel neighbourhood (group of elements that are considered closest to the central 
#             element in some measure space). The neighbourhood is defined by a 2-row Numpy matrix
#             where the first row contains the nerighbors row numbers and the second row the 
#             neighbours column numbers.the neighbourhood can have patterns s.a. horizontal, 
#             vertical, periodic, cross-like, etc. 
#    """
#    
#    def __init__(self,*arg):
#                
#        # kernel properties
#        self.k=np.mat([])
#        self.kdim = np.mat('0,0') 
#        self.kcenter = np.mat('0,0')
#        self.knhood = np.mat([])
#        
#        
#        if len(arg)==0:
#            return
#        elif len(arg)==2:
#            self.loadkernel(arg[0],arg[1])
#        elif len(arg)==3:
#             self.genkernel(arg[0],arg[1],arg[2])
#        
#    
#    def loadkernel(self,kernel,center):
#        """
#        loads a pre-defined kernel
#        inputs:
#        kernel: Numpy 2D matrix containing kernel values
#        center: 1 by 2 Numpy matrix containing the row and column numbers of the central element
#        """
#        self.k=kernel
#        self.kdim=np.mat(np.shape(self.k))
#        self.kcenter=center
#        ktemp=self.k; 
#        ktemp[center[0,0],center[0,1]]=0
#        self.knhood=np.mat(np.nonzero(ktemp))
#        
#    
#    def genkernel(self,dim,center,nhood):
#        """
#        generates the kernel object given the user-defined properties
#        inputs:
#        dim: 1 by 2 Numpy matrix containing kernel dimenstions
#        center: 1 by 2 Numpy matrix containing  the row and column numbers of the central element
#        nhood: Numpu 2D matrix, first row contains the neighbours row numbers and the second
#               row the neighbours column numbers
#        """
#        self.k=np.mat(np.zeros((dim[0,0],dim[0,1])))
#        self.k[center[0,0],center[0,1]]=1
#        self.k[nhood[0,:],nhood[1,:]]=1
#    
#    
#    def plotkernel(self):
#        """
#        plots the kernel matrix
#        """
#        plt.imshow(self.k,cmap='gray_r',interpolation='none')

        
        
        
        
        
        
        
        
        
        
        
        
   