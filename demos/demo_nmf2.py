import time
import numpy as np
import matplotlib.pylab as plt
import os

import nussl


def sine_example():
    if not os.path.exists(os.path.join('..', 'Output/')):
        os.mkdir(os.path.join('..', 'Output/'))

    # Make signals
    nSamples = 44100  # 1 second per each frequency

    sin1 = np.sin(np.linspace(0, 100 * 2 * np.pi, nSamples))  # Freq = 100 Hz
    sin2 = np.sin(np.linspace(0, 200 * 2 * np.pi, nSamples))  # Freq = 200 Hz
    sin3 = np.sin(np.linspace(0, 300 * 2 * np.pi, nSamples))  # Freq = 300 Hz

    sines = np.concatenate((sin1, sin2, sin3))

    # load into AudioSignal object and get stft
    signal = nussl.AudioSignal(audio_data_array=sines)
    signal.stft()
    stft = signal.get_stft_channel(1)

    # Start NMF and time it
    start = time.time()
    nmf = nussl.NMF(3)
    activation, dictionary = nmf.run()
    print '{0:.3f}'.format(time.time() - start), 'sec'

    # plot results
    plt.imshow(activation, interpolation='none', aspect='auto')
    ax = plt.axes()
    ya = ax.get_yaxis()
    ya.set_major_locator(plt.MaxNLocator(integer=True))
    xa = ax.get_xaxis()
    xa.set_major_locator(plt.MaxNLocator(integer=True))
    plt.title('Activation Matrix (H)')
    plt.savefig('../Output/A.png')
    plt.close()

    plt.imshow(dictionary, interpolation='none', aspect='auto')
    ax = plt.axes()
    ya = ax.get_yaxis()
    ya.set_major_locator(plt.MaxNLocator(integer=True))
    xa = ax.get_xaxis()
    xa.set_major_locator(plt.MaxNLocator(integer=True))
    plt.ylim([0, 20])
    plt.title('Template dictionary (W)')
    plt.savefig('../Output/B.png')


if __name__ == "__main__":
    sine_example()
