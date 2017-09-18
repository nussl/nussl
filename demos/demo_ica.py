import os
import sys
import matplotlib.pyplot as plt

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)

import numpy as np
from scipy import signal as scipy_signal
import nussl


def main():
    """
    This demo is adopted from the sklearn "BSS using FastICA" demo:
    http://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html

    """
    np.random.seed(0)
    sample_rate = nussl.DEFAULT_SAMPLE_RATE
    signal_duration = 1  # seconds
    num_samples = sample_rate * signal_duration
    time = np.linspace(0, signal_duration, num_samples)

    sig1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
    sig2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
    sig3 = scipy_signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

    full_signal = np.c_[sig1, sig2, sig3]
    full_signal += 0.2 * np.random.normal(size=full_signal.shape)  # Add noise
    full_signal /= full_signal.std(axis=0)  # Standardize data

    plt.plot(full_signal)
    plt.show()

    # Mix data
    mixing_matrix = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
    generated_observations = np.dot(full_signal, mixing_matrix.T)  # Generate observations
    plt.plot(generated_observations)
    plt.show()

    ica = nussl.ICA(observations_list=generated_observations)
    ica.run()
    sources = ica.make_audio_signals()
    estimated = []
    for i, s in enumerate(sources):
        s.write_audio_to_file('output/ica_src{}.wav'.format(i))
        estimated.append(s.get_channel(0))

    estimated = np.vstack(estimated).T

    plt.plot(estimated)
    plt.show()

    assert np.allclose(generated_observations, np.dot(estimated, ica.estimated_mixing_params.T) + ica.mean)

if __name__ == '__main__':
    main()
