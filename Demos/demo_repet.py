import AudioSignal
import Repet
import WindowAttributes
import os


def main():
    # input audio file
    inputName = '../Input/Sample1.wav'
    signal = AudioSignal.AudioSignal(pathToInputFile=inputName)

    if not os.path.exists('../Output/'):
        os.mkdir('../Output')

    # Set up window parameters
    win = WindowAttributes.WindowAttributes(signal.SampleRate)
    win.WindowLength = 2048
    win.WindowType = WindowAttributes.WindowType.HAMMING

    # Set up and run Repet
    repet = Repet.Repet(signal, Type=Repet.RepetType.SIM, windowAttributes=win)
    repet.MinDistanceBetweenFrames = 0.1
    repet.Run()

    # Get audio signals and write out to files
    bkgd, fgnd = repet.MakeAudioSignals()
    bkgd.WriteAudioFile('../Output/repet_background.wav')
    fgnd.WriteAudioFile('../Output/repet_foreground.wav')


if __name__ == '__main__':
    main()
