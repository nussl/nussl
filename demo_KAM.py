# In this demo the KAM algorithm is tested

import matplotlib.pyplot as plt
import numpy as np
from KAM import AudioSignal,kam 
from scikits.audiolab import wavread, wavwrite,play
import time    


# generate spectrograms of the separate sources 
WinL=4096

src1=AudioSignal('src1.wav')
src1.X=np.mat([])
src1.windowlength=WinL
src1.nfft=WinL
src1.makeplot=1; 
src1.fmaxplot=1000;

src2=AudioSignal('src2.wav')
src2.X=np.mat([])
src2.windowlength=WinL
src2.nfft=WinL
src2.makeplot=1; 
src2.fmaxplot=5000;

src3=AudioSignal('src3.wav')
src3.X=np.mat([])
src3.windowlength=WinL
src3.nfft=WinL
src3.makeplot=1; 
src3.fmaxplot=5000;

plt.figure(1)
plt.subplot(311)
src1.STFT()
plt.subplot(312)
src2.STFT()
plt.subplot(313)
src3.STFT()

src1=AudioSignal('src1.wav')
src1.X=np.mat([])
src1.windowlength=WinL
src1.nfft=WinL
src1.makeplot=1; 
src1.fmaxplot=1000;

# generate spectrograms of the mixture
WinL=4096

mix1=AudioSignal('mix2.wav')
mix1.X=np.mat([])
mix1.windowlength=WinL
mix1.nfft=WinL
mix1.makeplot=1; 
mix1.fmaxplot=5000;

plt.figure(2)
mix1.STFT()


######################## test with a very simple case #########################
# time signal
fs=44100
Ls=fs
ts=np.mat(np.arange(Ls)/float(fs))
x1=np.cos(2*np.pi*430.6640625*ts)
x2=np.zeros((1,Ls))
x2[0,11000:11050]=10
x2[0,22000:22050]=10
x3=(x1+x2).T

# spectrogram
WL=1024
aa=AudioSignal(x3,fs)
aa.X=np.mat([])
aa.makeplot=1
aa.fmaxplot=fs/2
aa.windowlength=WL
aa.nfft=WL
aa.overlapSamp=WL/2
TT=np.tile(aa.Tvec,(len(aa.Fvec),1))
FF=np.tile(aa.Fvec.T,(len(aa.Tvec),1)).T

plt.figure(3)
plt.subplot(211)
plt.plot(ts.T,x3)
plt.xlabel('t (sec)')
plt.ylabel('x(t)')
plt.title('Mixture')
plt.subplot(212)
aa.STFT()

# inputs of the 'kam' function
Inputfile=[x3,fs]
SourceKernels=[['horizontal',np.mat('88')],['vertical',np.mat('515')]]#,['cross',np.mat('5,5')]]
Numit=4


#FileName='dev1_nodrums_inst_mix.wav'
#FileName='mix2.wav'
#BlockLen=1 # length of each block of signal in seconds
#NB=int(np.floor(3/BlockLen)) # number of blocks in 4 seconds of signal
#Inputfile=[FileName,BlockLen,0]


start_time = time.clock()
shat,fhat=kam(Inputfile,SourceKernels,Numit)[0:2]
print time.clock() - start_time, "seconds"   



#for numblock in range(1,NB):
#   Inputfile=[FileName,BlockLen,BlockLen*numblock]
#   shat_temp=kam(Inputfile,SourceKernels,Numit)[0]
#   shat=np.append(shat,shat_temp,axis=0)      
#print time.clock() - start_time, "seconds"   


ss1=shat[:,0,0]
play(ss1,fs)

ss2=shat[:,0,1]
play(ss2,fs)


plt.figure(5)
plt.subplot(2,2,1)
plt.plot(ts.T,shat[:,:,0])
plt.ylabel(r'$\hat{s}_1$')
plt.xlabel('t(sec)')
plt.axis('tight')
plt.subplot(2,2,2)
plt.pcolormesh(TT,FF,np.log10(fhat[:,:,0]))
plt.ylabel(r'$\hat{f}_1$')
plt.axis('tight')
plt.ylim(aa.Fvec[0],5000)

plt.subplot(2,2,3)
plt.plot(ts.T,shat[:,0,1])
plt.ylabel(r'$\hat{s}_2$')
plt.xlabel('t(sec)')
plt.axis('tight')
plt.subplot(2,2,4)
plt.pcolormesh(TT,FF,np.log10(fhat[:,:,1]))
plt.ylabel(r'$\hat{f}_2$')
plt.axis('tight')
plt.ylim(aa.Fvec[0],5000)