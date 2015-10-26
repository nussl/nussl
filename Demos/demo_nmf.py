import AudioSignal, random, time
import numpy as np
import Nmf as NmfNU
import nimfa

random.seed(1)


def main():
    simpleExample()
    audioExample()


def simpleExample():
    """
    A simple example comparing NU NMF to nimfa
    :return:
    """
    print '-' * 60
    print ' ' * 19, 'SIMPLE EXAMPLE OUTPUT'
    print '-' * 60

    # Make two simple matrices
    n = 3
    a = np.arange(n ** 2).reshape((n, n))
    b = 2 * a + 3

    # Mix them together
    mixture = np.dot(b, a)

    # Set up NU NMF
    nBases = 2
    nmf = NmfNU.Nmf(mixture, nBases)
    nmf.shouldUseEpsilon = False
    nmf.maxNumIterations = 3000
    nmf.distanceMeasure = NmfNU.DistanceType.Divergence

    # Run NU NMF
    start = time.time()
    nmf.Run()
    print '{0:.3f}'.format(time.time() - start), 'seconds for NUNMF'

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
    i = 1


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
    firstFileName = 'Input/K0140.wav'
    secondFileName = 'Input/K0149.wav'

    firstNote = AudioSignal.AudioSignal(firstFileName)
    secondNote = AudioSignal.AudioSignal(secondFileName)

    # Combine notes into one file and save target
    bothNotesPre = firstNote + secondNote
    bothNotesPre.WriteAudioFile('Output/combined_preNMF.wav')

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
    start = time.time()
    nmf.Run()
    print int(time.time() - start), 'sec'

    # Make output files
    outFileNameBase = 'Output/NMFoutput_'
    i = 1
    newSignals = nmf.MakeAudioSignals()
    for signal in newSignals:
        outFileName = outFileNameBase + str(i) + '.wav'
        signal.WriteAudioFile(outFileName)
        i += 1

    # Recombine signals and make a new output file
    recombined = newSignals[0] + newSignals[1]
    recombined.WriteAudioFile('Output/combined_postNMF.wav')


if __name__ == '__main__':
    main()
