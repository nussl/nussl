import DUET
import AudioSignal
import WindowAttributes
from WindowType import WindowType


def main():
    # Load input file
    inputFileName = '../Input/dev1_female3_inst_mix.wav'
    signal = AudioSignal.AudioSignal(inputFileName=inputFileName)

    # set up FFT window attributes
    win = WindowAttributes.WindowAttributes(signal.SampleRate)
    win.WindowLength = 4096
    win.WindowType = WindowType.HAMMING
    win.WindowOverlap = 0.5 * win.WindowLength

    # Set up DUET algorithm and run
    duet = DUET.Duet(signal, aMin=-3, aMax=3, aNum=50, dMin=-3, dMax=3, dNum=50, threshold=0.2, aMinDistance=5,
                     dMinDistance=5, nSources=3, windowAttributes=win)
    duet.Run()
    duet.Plot('../Output/2d.png')
    duet.Plot('../Output/3d.png', three_d_plot=True)

    # Create output files
    outputFileStem = '../Output/duet_source'
    i = 1
    for s in duet.MakeAudioSignals():
        outputFileName = outputFileStem + str(i) + '.wav'
        s.WriteAudioFile(outputFileName)
        i += 1


if __name__ == '__main__':
    main()
