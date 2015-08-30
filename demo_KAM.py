# In this demo the KAM algorithm is tested

import matplotlib.pyplot as plt
import numpy as np
from KAM import AudioSignal,kam 
from scikits.audiolab import play,wavwrite
import time    

### Example (1): Test with a very simple scenario
###              The mixture includes a single frequency sinusoid (a horizontal line
###              in the TF domian) and two very short pulses (two vertical lines in 
###              the TF domain)

# time signal
fs=44100 # sampling rate
Ls=fs    # length of the signal (1 sec)
ts=np.mat(np.arange(Ls)/float(fs))
x1=np.cos(2*np.pi*430.6640625*ts) # sinusoid
x2=np.zeros((1,Ls))  
x2[0,11000:11050]=10 # first pulse
x2[0,22000:22050]=10 # second pulse
x3=(x1+x2).T

# spectrogram
WL=1024 
sig=AudioSignal(x3,fs)
sig.makeplot=1
sig.fmaxplot=fs/2
sig.windowlength=WL
sig.nfft=WL
sig.overlapSamp=WL/2


plt.figure(1)
plt.subplot(211)
plt.plot(ts.T,x3)
plt.xlabel('t (sec)')
plt.ylabel('x(t)')
plt.title('Mixture')
plt.subplot(212)
sig.STFT()
TT=np.tile(sig.Tvec,(len(sig.Fvec),1))
FF=np.tile(sig.Fvec.T,(len(sig.Tvec),1)).T

# inputs of the 'kam' function 
Inputfile=[x3,fs]
SourceKernels=[['horizontal',np.mat('10')],['vertical',np.mat('10')]] # 100ms horizontal and 387Hz vertical
SpecParam=np.zeros(1,dtype=[('windowlength',int),('overlapSamp',int),('nfft',int)])
SpecParam['windowlength']=WL
SpecParam['overlapSamp']=int(WL/2)    
SpecParam['nfft']=WL
Numit=2    

# call the kam function and record the running time
start_time = time.clock()
shat,fhat=kam(Inputfile,SourceKernels,Numit,SpecParam)[0:2]
print time.clock() - start_time, "seconds"   

# play the separated sources
ss1=shat[:,0,0]
play(ss1,fs)

ss2=shat[:,0,1]
play(ss2,fs)

# plot the separated time-domain signals and corresponding power spectral dencities
plt.figure(2)
plt.subplot(2,2,1)
plt.plot(ts.T,shat[:,:,0])
plt.ylabel(r'$\hat{s}_1(t)$')
plt.xlabel('t(sec)')
plt.axis('tight')
plt.subplot(2,2,2)
plt.pcolormesh(TT,FF,np.log10(fhat[:,:,0]))
plt.xlabel('t(sec)')
plt.ylabel('f(Hz)')
plt.title(r'$\hat{f}_1$')
plt.axis('tight')
plt.ylim(sig.Fvec[0],5000)

plt.subplot(2,2,3)
plt.plot(ts.T,shat[:,0,1])
plt.ylabel(r'$\hat{s}_2(t)$')
plt.xlabel('t(sec)')
plt.axis('tight')
plt.subplot(2,2,4)
plt.pcolormesh(TT,FF,np.log10(fhat[:,:,1]))
plt.xlabel('t(sec)')
plt.ylabel('f(Hz)')
plt.title(r'$\hat{f}_2$')
plt.axis('tight')
plt.ylim(sig.Fvec[0],5000)



### Example (2): Test with a mixture of music and speech
###              The mixture is composed of flute, kick drum and female voice
### Note: the second examples takes much longer to run than the first example

# generate spectrograms of the separate sources 
WinL=4096 # 93 ms window
Ovp=WinL/2 # 50% overlap  

src1=AudioSignal('src1.wav',3)
fs=src1.fs
src1.windowlength=WinL
src1.overlapSamp=Ovp
src1.nfft=WinL
src1.makeplot=1
src1.fmaxplot=1000

src2=AudioSignal('src2.wav',3)
src2.windowlength=WinL
src2.overlapSamp=Ovp
src2.nfft=WinL
src2.makeplot=1; 
src2.fmaxplot=2000;

plt.figure(3)
plt.subplot(211)
src1.STFT()
plt.title('Drum')
plt.subplot(212)
src2.STFT()
plt.title('Flute')


# generate spectrograms of the mixture
mix1=AudioSignal('mix4.wav')
mix1.windowlength=WinL
mix1.overlapSamp=Ovp
mix1.nfft=WinL
mix1.makeplot=1; 
mix1.fmaxplot=5000;

plt.figure(4)
mix1.STFT()
plt.title('Mixture')


# inputs of the 'kam' function
FileName='mix4.wav'
BlockLen=1.5 # length of each block of signal in seconds
AnalysisLen=3 # total length of the signal to be analyzed
NB=int(np.floor(AnalysisLen/BlockLen)) # total number of blocks 

Inputfile=[FileName,BlockLen,0]
#['vertical',np.mat([16])]
#['cross',np.mat([32,16])]
#['horizontal',np.mat([3])]

# define a horizontal kernel with Df>1
Df=3; Dt=20;
Nhood=lambda TFcoords1,TFcoords2: np.logical_and((np.abs(np.tile(TFcoords1[:,0],(1,TFcoords2.shape[0]))-np.tile(TFcoords2[:,0].T,(TFcoords1.shape[0],1)))<Df),\
                            (np.abs(np.tile(TFcoords1[:,1],(1,TFcoords2.shape[0]))-np.tile(TFcoords2[:,1].T,(TFcoords1.shape[0],1)))<Dt))

SourceKernels=[['periodic',np.mat([11,34])],['userdef',Nhood]]
SpecParam=np.zeros(1,dtype=[('windowlength',int),('overlapSamp',int),('nfft',int)])
SpecParam['windowlength']=WinL
SpecParam['overlapSamp']=Ovp
SpecParam['nfft']=WinL
Numit=5

# call the kam function and record the running time
start_time = time.clock()
shat,fhat=kam(Inputfile,SourceKernels,Numit,SpecParam)[0:2]
print time.clock() - start_time, "seconds"   

for numblock in range(1,NB):
   Inputfile=[FileName,BlockLen,BlockLen*numblock]
   shat_temp,fhat_temp=kam(Inputfile,SourceKernels,Numit,SpecParam)[0:2]
   shat=np.append(shat,shat_temp,axis=0)  
   fhat=np.append(fhat,fhat_temp,axis=1)    
print time.clock() - start_time, "seconds"   

# play the separated sources
ss1=shat[:,0,0]
play(ss1,fs)
wavwrite(ss1,'Source1_1.5s.wav',fs)

ss2=shat[:,0,1]
play(ss2,fs)
wavwrite(ss2,'Source2_1.5s.wav',fs)

# plot the separated time-domain signals and corresponding power spectral dencities
ts=np.mat(np.arange(shat.shape[0])/float(fs))
Fvec=mix1.Fvec
Tvec=mix1.Tvec[0:fhat.shape[1]]
TT=np.tile(Tvec,(len(Fvec),1))
FF=np.tile(Fvec.T,(len(Tvec),1)).T

plt.figure(5)
plt.subplot(2,2,1)
plt.plot(ts.T,src1.x[0:shat.shape[0]])
plt.plot(ts.T,shat[:,0,0])
plt.ylabel(r'$\hat{s}_1(t)$')
plt.axis('tight')
plt.subplot(2,2,2)
plt.pcolormesh(TT,FF,np.log10(fhat[:,:,0]))
plt.ylabel('f(Hz)')
plt.title(r'$\hat{f}_1$')
plt.axis('tight')
#plt.ylim(src1.Fvec[0],5000)

plt.subplot(2,2,3)
plt.plot(ts.T,src2.x[0:shat.shape[0]])
plt.plot(ts.T,shat[:,0,1])
plt.ylabel(r'$\hat{s}_2(t)$')
plt.axis('tight')
plt.subplot(2,2,4)
plt.pcolormesh(TT,FF,np.log10(fhat[:,:,1]))
plt.ylabel('f(Hz)')
plt.title(r'$\hat{f}_2$')
plt.axis('tight')
#plt.ylim(src2.Fvec[0],5000)


