import os
import nussl


def main():
    # input audio file
    input_name = os.path.join('..', 'Input','mix3.wav')
    signal = nussl.AudioSignal(path_to_input_file=input_name)

    # make a directory to store output if needed
    if not os.path.exists('../Output/'):
        os.mkdir('../Output')

    # Set up Repet
    repet = nussl.Repet(signal, do_mono=True)
    L = 2048
    repet.stft_params.window_length = L
    repet.stft_params.window_type = nussl.WindowType.HAMMING
    repet.stft_params.hop_length = L // 2
    repet.stft_params.n_fft_bins = L
    # and Run
    repet.run()

    # Get foreground and backgroun audio signals
    bkgd, fgnd = repet.make_audio_signals()

    # and write out to files
    bkgd.write_audio_to_file(os.path.join('..', 'Output', 'repet_background.wav'))
    fgnd.write_audio_to_file(os.path.join('..', 'Output', 'repet_foreground.wav'))

    # Set up Repet
    # repet = nussl.Repet(signal, repet_type=nussl.RepetType.SIM)
    # # and Run
    # # repet.run()
    #
    # # Get foreground and backgroun audio signals
    # bkgd, fgnd = repet.make_audio_signals()
    #
    # # and write out to files
    # bkgd.write_audio_to_file(os.path.join('..', 'Output', 'repet_background_SIM.wav'))
    # fgnd.write_audio_to_file(os.path.join('..', 'Output', 'repet_foreground_SIM.wav'))


if __name__ == '__main__':
    main()
