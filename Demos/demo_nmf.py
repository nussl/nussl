import random
import time

import numpy as np

import matplotlib.pylab as plt

import AudioSignal
import Nmf as NmfNU
import nimfa


# random.seed(1)


def main():
    # simpleExample()
    audioExample()
    # sineExample()


def simpleExample():
    """
    A simple example comparing NU NMF to nimfa
    :return:
    """
    print '-' * 60
    print ' ' * 19, 'SIMPLE EXAMPLE OUTPUT'
    print '-' * 60

    # Make two simple matrices
    n = 10
    a = np.arange(n ** 2).reshape((n, n))
    b = 2 * a + 3

    # Mix them together
    mixture = np.dot(b, a)

    # Set up NU NMF
    nBases = 2
    nmf = NmfNU.Nmf(mixture, nBases)
    nmf.shouldUseEpsilon = False
    nmf.maxNumIterations = 3000
    nmf.distanceMeasure = NmfNU.DistanceType.Euclidean

    # Run NU NMF
    start = time.time()
    nmf.Run()
    print '{0:.3f}'.format(time.time() - start), 'seconds for NUSSL'

    # Set up and run nimfa NMF
    nmf2 = nimfa.Nmf(mixture, update='divergence', rank=2, max_iter=nmf.maxNumIterations)
    start = time.time()
    nmf2_fit = nmf2()
    print '{0:.3f}'.format(time.time() - start), 'seconds for nimfa'

    # Get matrices from nimfa
    H = nmf2_fit.coef()
    W = nmf2_fit.basis()

    print '   ', '-' * 10, 'MIXTURES', '-' * 10

    print 'original mixture =\n', mixture
    print 'my mixture =\n', np.dot(nmf.templateVectors, nmf.activationMatrix)
    print 'nimfa mixture =\n', np.dot(W, H)

    print '    ', '-' * 10, 'NU NMF ', '-' * 10
    signals = nmf.RecombineCalculatedMatrices()
    for sig in signals:
        print sig

    print '    ', '-' * 10, 'NIMFA ', '-' * 10
    print H
    print W


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

    firstNote = AudioSignal.AudioSignal(firstFileName)
    secondNote = AudioSignal.AudioSignal(secondFileName)

    # Combine notes into one file and save target
    bothNotesPre = firstNote + secondNote
    bothNotesPre.WriteAudioFile('../Output/combined_preNMF.wav')

    bothNotesForNMF = firstNote + secondNote
    _, stft, _, _ = bothNotesForNMF.STFT()

    # Make some 'guesses'
    jitter = 0.2
    max = 2 ** 15
    for i, val in np.ndenumerate(firstNote.AudioData):
        firstNote.AudioData[i] += int(float(random.random() * jitter) * max)
    _, firstStft, _, _ = firstNote.STFT()
    firstGuessAct = np.sum(firstStft, axis=0)
    firstGuessVec = np.sum(firstStft, axis=1)

    for i, val in np.ndenumerate(secondNote.AudioData):
        secondNote.AudioData[i] += int(float(random.random() * jitter) * max)
    _, secondStft, _, _ = secondNote.STFT()
    secondGuessAct = np.sum(secondStft, axis=0)
    secondGuessVec = np.sum(secondStft, axis=1)

    # put them into activation matrix and template vectors
    GuessVec = np.array([firstGuessVec, secondGuessVec]).T
    GuessAct = np.array([firstGuessAct, secondGuessAct])

    # run NMF
    nmf = NmfNU.Nmf(stft, numNotes, activationMatrix=GuessAct, templateVectors=GuessVec)
    nmf.maxNumIterations = 100
    start = time.time()
    nmf.Run()
    print '{0:.3f}'.format(time.time() - start), 'seconds for NUSSL'

    nmf2 = nimfa.Nmf(stft, rank=numNotes, max_iter=nmf.maxNumIterations)
    start = time.time()
    nmf2()
    print '{0:.3f}'.format(time.time() - start), 'seconds for nimfa'

    # Make output files
    outFileNameBase = '../Output/NMFoutput_'
    i = 1
    newSignals = nmf.MakeAudioSignals()
    for signal in newSignals:
        outFileName = outFileNameBase + str(i) + '.wav'
        signal.WriteAudioFile(outFileName)
        i += 1

    # Recombine signals and make a new output file
    recombined = newSignals[0] + newSignals[1]
    recombined.WriteAudioFile('../Output/combined_postNMF.wav')


def sineExample():
    # Make signals
    nSamples = 44100  # 1 second per each frequency

    sin1 = np.sin(np.linspace(0, 100 * 2 * np.pi, nSamples))  # Freq = 100 Hz
    sin2 = np.sin(np.linspace(0, 200 * 2 * np.pi, nSamples))  # Freq = 200 Hz
    sin3 = np.sin(np.linspace(0, 300 * 2 * np.pi, nSamples))  # Freq = 300 Hz

    sines = np.concatenate((sin1, sin2, sin3))

    # load into AudioSignal object and get STFT
    signal = AudioSignal.AudioSignal(timeSeries=sines)
    _, stft, _, _ = signal.STFT()

    # Start NMF and time it
    start = time.time()
    nmf = NmfNU.Nmf(stft, 3)
    activation, dictionary = nmf.Run()
    print '{0:.3f}'.format(time.time() - start), 'sec'

    # Plot results
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
