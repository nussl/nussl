import AudioSignal, REPET_sim, pprint, SeparationBase, WindowType, FftUtils
from WindowType import WindowType


# import time

def main():
    #    print "starting"
    #myFile = 'Input/mix1.wav'
    #myAudioFile = AudioSignal.AudioSignal(myFile)
    #myAudioFile.windowType = WindowType.BLACKMAN
    #repet = REPET_sim.RepetSim(audioSignal=myAudioFile)
    #stems = repet.Run()

    wt = [WindowType.RECTANGULAR, WindowType.HAMMING, WindowType.HANNING, WindowType.BLACKMAN, WindowType.DEFAULT]
    for win in wt:
        w = FftUtils.MakeWindow(win, 100)
        print win, w


#    pprint.pprint(stems)
#    print "done"


if __name__ == '__main__':
    main()
