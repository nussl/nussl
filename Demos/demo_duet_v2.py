# In this demo the DUET algorithm is tested

from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt

plt.interactive('True')
import numpy as np
from SourceSeparation.FftUtils import f_stft
from nussl.DUET_v2 import duet_v2

raise DeprecationWarning('Don\'t get used to using this. It\'s going away soon!')

# close all the figure windows
plt.close('all')

# load the audio file
# fs,x = read('/Users/fpishdadian/SourceSeparation/Audio Samples/Input/dev1_female3_inst_mix.wav')  #speech,inst.
# fs,x = read('/Users/fpishdadian/SourceSeparation/Audio Samples/Input/dev1_female3_synthconv_130ms_5cm_mix.wav')  #speech,conv.
fs, x = read('Input/dev1_nodrums_inst_mix.wav')  # music,inst

# scale to -1.0 to 1.0
convert_16_bit = float(2 ** 15)
x = x / (convert_16_bit + 1.0)

x = np.mat(x).T
t = np.mat(np.arange(np.shape(x)[1]) / float(fs))

# generate and plot the spectrogram of the mixture
L = 4096
win = 'Hamming'
ovp = 0.5 * L
nfft = L
mkplot = 1
fmax = fs / 2

plt.figure(1)
plt.subplot(1, 2, 1)
plt.title('Mixture')
S, P, F, T = f_stft(x[0, :], L, win, ovp, fs, nfft, mkplot, fmax)


# compute and plot the 2D histogram of mixing parameters
a_min = -3
a_max = 3
a_num = 100
d_min = -3
d_max = 3
d_num = 100

sparam = np.array([(L, win, ovp, nfft, fs)],
                  dtype=[('winlen', int), ('wintype', '|S10'), ('overlap', int), ('numfreq', int), ('sampfreq', int)])
adparam = np.array([(a_min, a_max, a_num, d_min, d_max, d_num)],
                   dtype=[('amin', float), ('amax', float), ('anum', int), ('dmin', float), ('dmax', float),
                          ('dnum', int)])

xhat, ad_est, TFmask = duet_v2(x, sparam, adparam, 'y')
N = np.shape(xhat)[0]

# plot the mask in the same figure as the spectrogram
plt.figure(1)
plt.subplot(1, 2, 2)
TT = np.tile(T, (len(F[1:]), 1))
FF = np.tile(F[1:].T, (len(T), 1)).T
plt.pcolormesh(TT, FF, TFmask)
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('TF mask for ' + str(N) + ' source(s)')
plt.xlim(T[0], T[-1])
plt.ylim(F[0], F[-1])
plt.show()


# record the separated signals in .wav files
filePath = 'Output/duetv2OutSource'

for i in range(0, N):
    write(filePath + str(i + 1) + '.wav', fs, xhat[i, :])
