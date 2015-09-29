import AudioSignal, REPET_sim, pprint
#import time

def main():
    print "starting"
    myFile = 'Input/mix1.wav'
    myAudioFile = AudioSignal.AudioSignal(myFile)
    repet = REPET_sim.RepetSim(audioSignal=myAudioFile)
    stems = repet.Run()
    pprint.pprint(stems)
    print "done"


if __name__ == '__main__':
    main()
