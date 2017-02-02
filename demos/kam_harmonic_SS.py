# in this script a mixture of harmonic sources will be analyzed for separation uisng KAM
# along with harmonic kernels

import matplotlib.pyplot as plt
plt.interactive('True')
import numpy as np
from nussl.KAM import AudioSignal,kam
import time  

# close all figure windows
plt.close('all')

# load the audio mixture and generate spectrograms

FileName='/Users/fpishdadian/SourceSeparation/Audio Samples/Input/piano_mix2.wav'
mix=AudioSignal(FileName)

WinL=2*2048 # 93 ms window
Ovp=3*WinL/4 # 50% overlap   
mix.windowlength=WinL
mix.overlap_samples=Ovp
mix.num_fft_bins=WinL
mix.makeplot=1; 
mix.fmaxplot=5000;

plt.figure(1)
mix.do_STFT()
plt.title('Mixture')


# inputs of the 'kam' function

Inputfile=[FileName,'full length',0]

# define harmonic kernels with different fundamental freq.s
Np=1
SourceKernels=[]
SourceKernels.append(['harmonic',np.mat([24,Np])])  # source #1
SourceKernels.append(['harmonic',np.mat([32,Np])])  # source #2 
SourceKernels.append(['harmonic',np.mat([36,Np])])  # source #3 
SourceKernels.append(['harmonic',np.mat([40,Np])])  # source #4 

# define smooth harmonic kernels with differnt fundamental freq.s
#SourceKernels=[]
#P=np.array([24,32,36,40])
#NP=1
#Df=NP*P+1 
#def ismember(A, B):
#    return np.sum(np.array([ i == B.T for i in A.T ]),axis=1).T
#for i in range(0,4):    
#   Nhood=lambda TFcoords1,TFcoords2: np.logical_and(np.logical_and((np.tile(TFcoords1[:,1],(1,TFcoords2.shape[0]))==np.tile(TFcoords2[:,1].T,(TFcoords1.shape[0],1))),\
#                        (np.abs(np.tile(TFcoords1[:,0],(1,TFcoords2.shape[0]))-np.tile(TFcoords2[:,0].T,(TFcoords1.shape[0],1)))<Df[i])),\
#                        (ismember(np.mod(np.tile(TFcoords1[:,0],(1,TFcoords2.shape[0]))-np.tile(TFcoords2[:,0].T,(TFcoords1.shape[0],1)),P[i]),np.mat([0,1,P[i]-1])))) 
#   SourceKernels.append(['userdef',Nhood])
   

SpecParams=np.zeros(1,dtype=[('windowlength',int),('overlap_samples',int),('num_fft_bins',int)])
SpecParams['windowlength']=WinL
SpecParams['overlap_samples']=Ovp
SpecParams['num_fft_bins']=WinL

Numit=3

# call the kam function and record the running time
start_time = time.clock()
shat,fhat=kam(Inputfile,SourceKernels,Numit=Numit,SpecParams=SpecParams,FullKernel=False)[0:2]
print time.clock() - start_time, "seconds"   

# write the separated sources to .wav files
Ns=len(SourceKernels)
OutPath='/Users/fpishdadian/SourceSeparation/Audio Samples/Output/'
for i in range(0,Ns):
    ssi=AudioSignal(audiosig=shat[:,:,i],fs=mix.fs)  
    ssi.writeaudiofile(OutPath+'kamHout'+str(i+1)+'.wav')
    
 
 # plot the separated time-domain signals and corresponding power spectral dencities
ts=np.mat(np.arange(shat.shape[0])/float(mix.fs))
Fvec=mix.freq_vec
Tvec=mix.time_vec[0:fhat.shape[1]]
TT=np.tile(Tvec,(len(Fvec),1))
FF=np.tile(Fvec.T,(len(Tvec),1)).T


plt.figure(2)
for i in range(0,Ns):
    plt.subplot(Ns,1,i+1)
    #plt.plot(ts.T,src1.x[0:shat.shape[0]])
    plt.plot(ts.T,shat[:,0,i])
    plt.ylabel(r'$\hat{s}_'+str(i+1)+'(t)$')
    plt.axis('tight')
plt.xlabel('t(s)')    
    
plt.figure(3)
for i in range(0,Ns):    
    plt.subplot(Ns,1,i+1)
    plt.pcolormesh(TT,FF,np.log10(fhat[:,:,i]))
    plt.ylabel('f(Hz)')
    plt.title(r'$\hat{f}_'+str(i+1)+' $')
    plt.axis('tight')
    #plt.ylim(src1.freq_vec[0],5000)
plt.xlabel('t(s)')     
    
