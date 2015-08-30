# In this demo the original REPET-SIM algorithm is tested

import matplotlib.pyplot as plt
import numpy as np
from scikits.audiolab import wavread
from scikits.audiolab import play
from f_stft import f_stft
from REPET_sim import repet_sim

# close all figure windows
plt.close('all')

# load the audio file
x,fs,enc = wavread('Sample1.wav');
#x,fs,enc = wavread('mix4.wav')
 
x=np.mat(x)
t=np.mat(np.arange(np.shape(x)[1])/float(fs))
play(x,fs)

# generate and plot the spectrogram of the mixture
L=2048;
win='Hamming'
ovp=0.5*L
nfft=L
mkplot=1
fmax=5000;

plt.figure(1)
plt.title('Mixture');
Sm=f_stft(np.mat(x),L,win,ovp,fs,nfft,mkplot,fmax); 
plt.show()

# separation
par=np.array([0,0.01,10]); 
y_sim = repet_sim(np.mat(x),fs,par)

# play and plot the background and foreground
plt.figure(3)
plt.subplot(2,1,1)
plt.title('Background time-domain signal');
plt.plot(t.T,y_sim.T)
plt.axis('tight')
plt.show()
plt.subplot(2,1,2)
plt.title('Foreground time-domain signal');
plt.plot(t.T,(x-y_sim).T)
plt.axis('tight')
plt.show()

plt.figure(4)
plt.subplot(2,1,1)
plt.title('Background Spectrogram');
Sb=f_stft(np.mat(y_sim),L,win,ovp,fs,nfft,mkplot,fmax); 
plt.show()
plt.subplot(2,1,2)
plt.title('Foreground Spectrogram');
Sf=f_stft(np.mat(x-y_sim),L,win,ovp,fs,nfft,mkplot,fmax); 
plt.show()

play(y_sim,fs)   # background
play(x-y_sim,fs)  # foreground

# check whether the separated spectrograms add up to the original spectrogram
Spec_diff=np.abs(Sm[0] - (Sb[0]+Sf[0]))

if Spec_diff.max()<1e-10:
    print('Background and foreground add up to the origianl mixture.')