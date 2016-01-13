# In this demo the RPCA-based source separation method is tested

from scipy.io.wavfile import read,write
import matplotlib.pyplot as plt
plt.interactive('True')
import numpy as np
from f_stft import f_stft
from nussl.RPCA import rpca_ss
import time  

# close all figure windows
plt.close('all')

# load the audio file
#fs,x = read('/Users/fpishdadian/SourceSeparation/Audio Samples/Input/Sample2.wav')
fs,x = read('/Users/fpishdadian/SourceSeparation/Audio Samples/Input/mix5.wav')
x=np.array(x,ndmin=2)

# scale to -1.0 to 1.0
convert_16_bit = float(2**15)
x = x / (convert_16_bit + 1.0)

t=np.mat(np.arange(np.shape(x)[1])/float(fs))

# generate and plot the spectrogram of the mixture   
L=2048
win='Hamming'
ovp=0.75*L
nfft=L
mkplot=1
fmax=fs/2

plt.figure(1)
plt.title('Mixture')
Sm = f_stft(np.mat(x),L,win,ovp,fs,nfft,mkplot,fmax) 

# spactrogram parameters
specparam = [L,'Hamming',ovp,nfft]

# separation
start_time = time.clock()
y_f,y_b = rpca_ss(x,fs,specparam,mask=True,maskgain=1)
print time.clock() - start_time, "seconds"  


# normalize output signals
fg=y_f/np.abs(y_f).max()
bg=y_b/np.abs(y_b).max()

# plot the background and foreground
plt.figure(3)
plt.subplot(2,1,1)
plt.title('Background time-domain signal')
plt.plot(t.T,bg.T)
plt.axis('tight')
plt.show()
plt.subplot(2,1,2)
plt.title('Foreground time-domain signal')
plt.plot(t.T,fg.T)
plt.axis('tight')
plt.show()

plt.figure(4)
plt.subplot(2,1,1)
plt.title('Background Spectrogram')
Sb = f_stft(np.mat(y_b),L,win,ovp,fs,nfft,mkplot,fmax) 
plt.show()
plt.subplot(2,1,2)
plt.title('Foreground Spectrogram')
Sf = f_stft(np.mat(y_f),L,win,ovp,fs,nfft,mkplot,fmax) 
plt.show()

# check whether the separated spectrograms add up to the original spectrogram
Spec_diff=np.abs(Sm[0] - (Sb[0]+Sf[0]))

if Spec_diff.max()<1e-10:
    print('Background and foreground add up to the origianl mixture.')
    
# record the separated background and foreground in .wav files
filePath='/Users/fpishdadian/SourceSeparation/Audio Samples/Output/'
write(filePath+'rpcaBackground.wav',fs,bg.T)
write(filePath+'rpcaForeground.wav',fs,fg.T)