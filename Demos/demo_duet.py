# In this demo the DUET algorithm is tested

from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt

plt.interactive('True')
import numpy as np
from FftUtils import f_stft
from DUET import duet

# close all the figure windows
plt.close('all')

# load the audio file
fs, x = read('/Users/fpishdadian/SourceSeparation/Audio Samples/Input/dev1_female3_inst_mix.wav')  # speech,inst.
# fs,x = read('/Users/fpishdadian/SourceSeparation/Audio Samples/Input/dev1_female3_synthconv_130ms_5cm_mix.wav')  #speech,conv.
# fs,x = read('/Users/fpishdadian/SourceSeparation/Audio Samples/Input/dev1_nodrums_inst_mix.wav')  #music,inst

# scale to -1.0 to 1.0
convert_16_bit = float(2 ** 15)
x = x / (convert_16_bit + 1.0)

x = np.mat(x).T
t = np.mat(np.arange(np.shape(x)[1]) / float(fs))


# generate and plot the spectrogram of the mixture
L = 4096;
win = 'Hamming'
ovp = 0.5 * L
nfft = L
mkplot = 1
fmax = 3000

plt.figure()
X = f_stft(x[0, :], L, win, ovp, fs, nfft, mkplot, fmax)
plt.title('Mixture')

# compute and plot the 2D histogram of mixing parameters
a_min = -3;
a_max = 3;
a_num = 50
d_min = -3;
d_max = 3;
d_num = 50

sparam = np.array([(L, win, ovp, nfft, fs)],
                  dtype=[('winlen', int), ('wintype', '|S10'), ('overlap', int), ('numfreq', int), ('sampfreq', int)])
adparam = np.array([(a_min, a_max, a_num, d_min, d_max, d_num)], dtype=[('amin', float), ('amax', float)
    , ('anum', int), ('dmin', float), ('dmax', float), ('dnum', int)])

thr = 0.2;
a_mindist = 5;
d_mindist = 5;
N = 3
Pr = np.array([thr, a_mindist, d_mindist, N])

xhat, ad_est = duet(x, sparam, adparam, Pr, 'y')


# record the separated signals in .wav files
filePath = '/Users/fpishdadian/SourceSeparation/Audio Samples/Output/duetOutSource'

for i in range(0, N):
    write(filePath + str(i + 1) + '.wav', fs, xhat[i, :].T)
