# In this demo the AudioMix module will be tested. AudioMix can be used to synthesize 
# convolutive mixtures.

from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt

plt.interactive('True')
import numpy as np
from nussl.AudioMix import mkmixture

# close all figure windows
plt.close('all')

# load single channel signals

InputPath = '/Users/fpishdadian/SourceSeparation/Audio Samples/Input/'
fs, s1 = read(InputPath + 'K0140.wav')
fs, s2 = read(InputPath + 'K0144.wav')
fs, s3 = read(InputPath + 'K0147.wav')

# normalize
s1 = s1 / float(np.abs(s1).max())
s2 = s2 / float(np.abs(s2).max())
s3 = s3 / float(np.abs(s3).max())

Ls = np.min([s1.size, s2.size, s3.size])
ts = (1 / float(fs)) * np.arange(Ls)

Sources = np.vstack([s1[0:Ls], s2[0:Ls], s3[0:Ls]])

N = 3  # number of sources
Sn = Sources[0:N, :]

# specify the locations of the sources and mics and room parameters
Center = np.array([2, 2, 1])
TH = np.array([np.pi / 4, 0, -np.pi / 4])
CosTH = np.cos(TH)
SinTH = np.sin(TH)
SCenter = np.tile(Center, (N, 1))
Ps = np.array([SCenter[:, 0] + CosTH[0:N], SCenter[:, 1] + SinTH[0:N], SCenter[:, 2]]).T  # source coordinates

M = 2  # number of mics (channel mixtures)
MicPos = np.array([0.025, -0.025])
MCenter = np.tile(Center, (M, 1))
Pm = np.array([MCenter[:, 0], MCenter[:, 1] + MicPos, MCenter[:, 2]]).T  # mic coordinates

numvir = 1
refcoeff = 1
RoomParams = np.array([numvir, refcoeff])
RoomDim = np.array([5, 4, 3])

mixparam = [Ps, Pm, RoomParams, RoomDim]

Mixtures, SourceIMS, H = mkmixture(Sn, mixparam, fs, rsch=True)

hh = H[0][0]
th = (1. / fs) * np.arange(hh.shape[1])
th.shape = hh.shape
plt.figure(2)
plt.plot(th.T, hh.T)
plt.xlabel('time(s)')

OutputPath = '/Users/fpishdadian/SourceSeparation/Audio Samples/Output/'
write(OutputPath + 'convmix.wav', fs, Mixtures.T)
