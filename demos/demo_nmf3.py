import random
import time
import numpy as np
import os

import nussl


def audio_example():
    """
    A simple source separation with audio files. Inputs two files of piano notes
    and adds them together and then creates two "guesses" for NMF to start from.
    Outputs files
    :return:
    """

    if not os.path.exists(os.path.join('..', 'Output/')):
        os.mkdir(os.path.join('..', 'Output/'))

    numNotes = 2

    # Two input files
    firstFileName = os.path.join('..', 'Input', 'K0140.wav')
    secondFileName = os.path.join('..', 'Input', 'K0149.wav')

    firstNote = nussl.AudioSignal(firstFileName)
    secondNote = nussl.AudioSignal(secondFileName)

    # Combine notes into one file and save target
    bothNotesPre = firstNote + secondNote
    bothNotesPre.write_audio_to_file(os.path.join('..', 'Output', 'combined_preNMF.wav'))

    bothNotesForNMF = firstNote + secondNote
    _, stft, _, _ = bothNotesForNMF.stft()

    # Make some 'guesses'
    jitter = 0.2
    max = 2 ** 15
    for i, val in np.ndenumerate(firstNote.audio_data):
        firstNote.audio_data[i] += int(float(random.random() * jitter) * max)
    _, firstStft, _, _ = firstNote.stft()
    firstGuessAct = np.sum(firstStft, axis=0)
    firstGuessVec = np.sum(firstStft, axis=1)

    for i, val in np.ndenumerate(secondNote.audio_data):
        secondNote.audio_data[i] += int(float(random.random() * jitter) * max)
    _, secondStft, _, _ = secondNote.stft()
    secondGuessAct = np.sum(secondStft, axis=0)
    secondGuessVec = np.sum(secondStft, axis=1)

    # put them into activation matrix and template vectors
    GuessVec = np.array([firstGuessVec, secondGuessVec]).T
    GuessAct = np.array([firstGuessAct, secondGuessAct])

    # run NMF
    nmf = nussl.NMF(numNotes)
    nmf.max_num_iterations = 100
    start = time.time()
    nmf.run()
    print '{0:.3f}'.format(time.time() - start), 'seconds for NUSSL'

    # Make output files
    outFileNameBase = os.path.join('..', 'Output', 'NMFoutput_')
    i = 1
    newSignals = nmf.make_audio_signals()
    for signal in newSignals:
        outFileName = outFileNameBase + str(i) + '.wav'
        signal.write_audio_to_file(outFileName)
        i += 1

    # Recombine signals and make a new output file
    recombined = newSignals[0] + newSignals[1]
    recombined.write_audio_to_file(os.path.join('..', 'Output', 'combined_postNMF.wav'))


if __name__ == "__main__":
    audio_example()
