from nussl import AudioSignal, Repet


def main():
    myFile = 'Input/K0149.wav'
    myAudioFile = AudioSignal(myFile)
    repet = Repet(audioSignal=myAudioFile)
    # FftUtils.PlotStft(myAudioFile)
    stems = repet()

    # for stem in stems:
    #    stem.Write


if __name__ == '__main__':
    main()
