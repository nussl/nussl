import random
import time

import numpy as np
import matplotlib.pylab as plt

import nussl


def main():
    # simpleExample()
    # audioExample()
    sineExample()


def simpleExample():
    """
    A simple example comparing NU NMF to nimfa
    :return:
    """
    print '-' * 60
    print ' ' * 19, 'SIMPLE EXAMPLE OUTPUT'
    print '-' * 60

    # Make two simple matrices
    n = 4
    a = np.arange(n ** 2).reshape((n, n))
    b = 2 * a + 3

    # Mix them together
    mixture = np.dot(b, a)

    # Set up NU NMF
    nBases = 2
    nmf = nussl.Nmf(mixture, nBases)
    nmf.should_use_epsilon = False
    nmf.max_num_iterations = 3000
    nmf.distance_measure = nussl.DistanceType.EUCLIDEAN

    # run NU NMF
    start = time.time()
    nmf.run()
    print '{0:.3f}'.format(time.time() - start), 'seconds for NUSSL'

    print 'original mixture =\n', mixture
    print 'my mixture =\n', np.dot(nmf.templates, nmf.activation_matrix)

    print '    ', '-' * 10, 'NU NMF ', '-' * 10
    signals = nmf.recombine_calculated_matrices()
    for sig in signals:
        print sig


def audioExample():
    """
    A simple source separation with audio files. Inputs two files of piano notes
    and adds them together and then creates two "guesses" for NMF to start from.
    Outputs files
    :return:
    """

    print '-' * 60
    print ' ' * 19, 'AUDIO EXAMPLE OUTPUT'
    print '-' * 60

    numNotes = 2

    # Two input files
    firstFileName = '../Input/K0140.wav'
    secondFileName = '../Input/K0149.wav'

    firstNote = nussl.AudioSignal(firstFileName)
    secondNote = nussl.AudioSignal(secondFileName)

    # Combine notes into one file and save target
    bothNotesPre = firstNote + secondNote
    bothNotesPre.write_audio_to_file('../Output/combined_preNMF.wav')

    bothNotesForNMF = firstNote + secondNote
    _, stft, _, _ = bothNotesForNMF.do_STFT()

    # Make some 'guesses'
    jitter = 0.2
    max = 2 ** 15
    for i, val in np.ndenumerate(firstNote.audio_data):
        firstNote.audio_data[i] += int(float(random.random() * jitter) * max)
    _, firstStft, _, _ = firstNote.do_STFT()
    firstGuessAct = np.sum(firstStft, axis=0)
    firstGuessVec = np.sum(firstStft, axis=1)

    for i, val in np.ndenumerate(secondNote.audio_data):
        secondNote.audio_data[i] += int(float(random.random() * jitter) * max)
    _, secondStft, _, _ = secondNote.do_STFT()
    secondGuessAct = np.sum(secondStft, axis=0)
    secondGuessVec = np.sum(secondStft, axis=1)

    # put them into activation matrix and template vectors
    GuessVec = np.array([firstGuessVec, secondGuessVec]).T
    GuessAct = np.array([firstGuessAct, secondGuessAct])

    # run NMF
    nmf = nussl.Nmf(stft, numNotes, activation_matrix=GuessAct, templates=GuessVec)
    nmf.max_num_iterations = 100
    start = time.time()
    nmf.run()
    print '{0:.3f}'.format(time.time() - start), 'seconds for NUSSL'

    # Make output files
    outFileNameBase = '../Output/NMFoutput_'
    i = 1
    newSignals = nmf.make_audio_signals()
    for signal in newSignals:
        outFileName = outFileNameBase + str(i) + '.wav'
        signal.write_audio_to_file(outFileName)
        i += 1

    # Recombine signals and make a new output file
    recombined = newSignals[0] + newSignals[1]
    recombined.write_audio_to_file('../Output/combined_postNMF.wav')


def sineExample():
    # Make signals
    nSamples = 44100  # 1 second per each frequency

    sin1 = np.sin(np.linspace(0, 100 * 2 * np.pi, nSamples))  # Freq = 100 Hz
    sin2 = np.sin(np.linspace(0, 200 * 2 * np.pi, nSamples))  # Freq = 200 Hz
    sin3 = np.sin(np.linspace(0, 300 * 2 * np.pi, nSamples))  # Freq = 300 Hz

    sines = np.concatenate((sin1, sin2, sin3))

    # load into AudioSignal object and get do_STFT
    signal = nussl.AudioSignal(audio_data_array=sines)
    _, stft, _, _ = signal.do_STFT()

    # Start NMF and time it
    start = time.time()
    nmf = nussl.Nmf(stft, 3)
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


if __name__ == '__main__':
    main()
