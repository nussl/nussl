# In this demo the DUET algorithm is tested

import matplotlib.pyplot as plt
import numpy as np
from scikits.audiolab import wavread
from scikits.audiolab import play
from f_stft import f_stft
from DUET import duet

# load the audio file
#x,fs,enc = wavread('dev1_female3_inst_mix.wav'); 
x,fs,enc = wavread('dev1_nodrums_inst_mix.wav'); 
x=np.mat(x).T
t=np.mat(np.arange(np.shape(x)[1])/float(fs))
#play(x[0,:],fs)

# generate and plot the spectrogram of the mixture
L=4*1024;
win='Hamming'
ovp=0.5*L
nfft=L
mkplot=1
fmax=3000

plt.figure(1)
plt.title('Mixture');
X = f_stft(x[0,:],L,win,ovp,nfft,fs,mkplot,fmax); 
plt.show()

# compute and plot the 2D histogram of mixing parameters
a_min=-3; a_max=3; a_num=50;
d_min=-3; d_max=3; d_num=50;

sparam = np.array([(L,win,ovp,nfft,fs)],dtype=[('winlen',int),('wintype','|S10'),('overlap',int),('numfreq',int),('sampfreq',int)])
adparam = np.array([(a_min,a_max,a_num,d_min,d_max,d_num)],dtype=[('amin',float),('amax',float)
,('anum',int),('dmin',float),('dmax',float),('dnum',int)])

thr=0.1; a_mindist=5; d_mindist=5; N=3;
Pr=np.array([0.1,5,5,3])

xhat,ad_est=duet(x,sparam,adparam,Pr,'y')

# play the separated sources
#for i in range(0,N):
#  play(xhat[i,:],fs)
 








