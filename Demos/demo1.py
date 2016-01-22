from nussl import AudioSignal, Repet


def main():
    #TODO: This don't work...
    myFile = '../Input/K0149.wav'
    myAudioFile = AudioSignal(myFile)
    repet = Repet(audio_signal=myAudioFile)
    # FftUtils.PlotStft(myAudioFile)
    stems = repet()

    # for stem in stems:
    #    stem.Write


if __name__ == '__main__':
    main()
