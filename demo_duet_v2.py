# In this demo the DUET algorithm is tested

import matplotlib.pyplot as plt
import numpy as np
from scikits.audiolab import wavread
from scikits.audiolab import play
from f_stft import f_stft
from DUET_v2 import duet_v2

# load the audio file
#x,fs,enc = wavread('dev1_female3_inst_mix.wav');  #speech,inst.
x,fs,enc = wavread('dev1_female3_synthconv_130ms_5cm_mix.wav');  #speech,conv.

#x,fs,enc = wavread('dev1_nodrums_inst_mix.wav');  #music,inst

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
a_min=-3; a_max=3; a_num=100;
d_min=-3; d_max=3; d_num=100;

sparam = np.array([(L,win,ovp,nfft,fs)],dtype=[('winlen',int),('wintype','|S10'),('overlap',int),('numfreq',int),('sampfreq',int)])
adparam = np.array([(a_min,a_max,a_num,d_min,d_max,d_num)],dtype=[('amin',float),('amax',float)
,('anum',int),('dmin',float),('dmax',float),('dnum',int)])

xhat,ad_est=duet_v2(x,sparam,adparam,'y')

# play the separated sources
N=np.shape(xhat)[0]
for i in range(0,N):
  play(xhat[i,:],fs)
 








