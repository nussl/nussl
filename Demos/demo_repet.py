import AudioSignal
import Repet
import WindowAttributes


def main():
    # input audio file
    inputName = '../Input/Sample1.wav'
    signal = AudioSignal.AudioSignal(inputFileName=inputName)

    # Set up window parameters
    win = WindowAttributes.WindowAttributes(signal.SampleRate)
    win.WindowLength = 2048
    win.WindowType = WindowAttributes.WindowType.HAMMING
    win.WindowOverlap = 1024
    win.Nfft = 2048

    # Set up and run Repet
    # repet = Repet.Repet(signal, Type=Repet.RepetType.SIM, windowAttributes=win)
    # repet.MinDistanceBetweenFrames = 0.1
    # repet.Run()
    #
    # # Get audio signals and write out to files
    # bkgd, fgnd = repet.MakeAudioSignals()
    # bkgd.WriteAudioFile('../Output/repet_background.wav')
    # fgnd.WriteAudioFile('../Output/repet_foreground.wav')

    repet2 = Repet.Repet(signal, Type=Repet.RepetType.ORIGINAL, windowAttributes=win)
    repet2.Run()


if __name__ == '__main__':
    main()
