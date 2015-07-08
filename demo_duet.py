# In this demo the DUET algorithm is tested

import matplotlib.pyplot as plt
import numpy as np
from scikits.audiolab import wavread
from scikits.audiolab import play
from f_stft import f_stft
from DUET import duet

# load the audio file
x,fs,enc = wavread('dev1_female3_inst_mix.wav'); 
x=np.mat(x).T
t=np.mat(np.arange(np.shape(x)[1])/float(fs))
#play(x[0,:],fs)

# generate and plot the spectrogram of the mixture
L=4*1024;
win='Hamming'
ovp=0.5*L
nfft=L
mkplot=1
fmax=3000;

plt.figure(1)
plt.title('Mixture');
Sm = f_stft(x[0,:],L,win,ovp,nfft,fs,mkplot,fmax); 
plt.show()

# compute and plot the 2D histogram of mixing parameters
a_min=-3; a_max=3; a_num=50;
d_min=-3; d_max=3; d_num=50;

sparam = np.array([L,win,ovp,nfft,fs])
adparam = np.array([-3,3,50,-3,3,50])
Pr=np.array([1,2,3])

hist=duet(x,sparam,adparam,Pr)