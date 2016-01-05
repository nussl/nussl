import AudioSignal
import Repet


# import time

def main():
    print "starting"
    myFile = 'Input/K0149.wav'
    myAudioFile = AudioSignal.AudioSignal(myFile)
    repet = Repet.Repet(audioSignal=myAudioFile)
    # FftUtils.PlotStft(myAudioFile)
    stems = repet()

    # for stem in stems:
    #    stem.Write


if __name__ == '__main__':
    main()
