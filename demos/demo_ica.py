import os
import sys
import matplotlib.pyplot as plt

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)

import numpy as np
from scipy import signal
from nussl import AudioSignal, ICA

np.random.seed(0)
n_samples = 44100*5
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal


S = np.c_[s1, s2, s3]

S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data

plt.plot(S)
plt.show()
# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations
plt.plot(X)
plt.show()
observations = []

for i in range(X.shape[1]):
    observations.append(AudioSignal(audio_data_array=X[:, i], sample_rate = 44100))

observations = ICA.transform_observations_to_audio_signal(observations)

ica = ICA(input_audio_signal=observations)
ica.run()
sources = ica.make_audio_signals()
estimated = []
for i,s in enumerate(sources):
    s.write_audio_to_file('output/ICA %d.wav' % i)
    estimated.append(s.get_channel(0))

estimated = np.vstack(estimated).T

plt.plot(estimated)
plt.show()

assert np.allclose(X, np.dot(estimated, ica.mixing.T) + ica.mean)