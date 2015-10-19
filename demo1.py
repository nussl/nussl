import AudioSignal, REPET_sim, pprint, SeparationBase, WindowType
from WindowType import WindowType


# import time

def main():
    #    print "starting"
    myFile = 'Input/mix1.wav'
    myAudioFile = AudioSignal.AudioSignal(myFile)
    myAudioFile.windowType = WindowType.BLACKMAN
    repet = REPET_sim.RepetSim(audioSignal=myAudioFile)
    stems = repet.Run()


#    pprint.pprint(stems)
#    print "done"


if __name__ == '__main__':
    main()
