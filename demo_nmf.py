import AudioSignal
from Nmf import Nmf


def main():
    myFileName = 'Input/mix1.wav'
    myAudioFile = AudioSignal.AudioSignal(myFileName)
    _, stft, _, _ = myAudioFile.STFT()

    nBases = 10

    nmf = Nmf(stft, nBases)
    activation, bases = nmf.Run()

    outFileNameBase = 'output_'
    i = 1
    for signal in nmf.MakeAudioSignals():
        outFileName = outFileNameBase + str(i) + '.wav'
        signal.WriteAudioFile(outFileName)
        i += 1


if __name__ == '__main__':
    main()
