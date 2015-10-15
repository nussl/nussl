# In this demo the RPCA-based source separation method is tested

from scipy.io.wavfile import read,write
import matplotlib.pyplot as plt
plt.interactive('True')
import numpy as np
from f_stft import f_stft
from REPET_org import repet
import time  

# close all figure windows
plt.close('all')

# load the audio file
fs,x = read('/Users/fpishdadian/SourceSeparation/Audio Samples/Input/Sample1.wav')
x=np.array(x,ndmin=2)

# scale to -1.0 to 1.0
x=x/np.abs(x).max()

# generate and plot the spectrogram of the mixture   
L=2048
win='Hamming'
ovp=0.5*L
nfft=L
mkplot=1
fmax=5000

plt.figure(1)
plt.title('Mixture')
Sm = f_stft(np.mat(x),L,win,ovp,fs,nfft,mkplot,fmax) 

# separation
#start_time = time.clock()
#y_f,y_b = repet(np.mat(x),fs)
#print time.clock() - start_time, "seconds"  