import os
from nussl import Duet, AudioSignal


def main():
    # Load input file
    input_file_name = '../Input/dev1_female3_inst_mix.wav'
    signal = AudioSignal(path_to_input_file=input_file_name)

    # make a directory to store output if needed
    if not os.path.exists(os.path.join('..', 'Output/')):
        os.mkdir(os.path.join('..', 'Output/'))

    # Set up DUET
    duet = Duet(signal, 3)
    # duet.stft_params.hop_length = 200
    # and run
    duet.run()

    # plot histogram results
    duet.plot('../Output/2d.png')
    duet.plot('../Output/3d.png', three_d_plot=True)

    # Create output file for each source found
    output_name_stem = '../Output/duet_source'
    i = 1
    for s in duet.make_audio_signals():
        output_file_name = output_name_stem + str(i) + '.wav'
        s.write_audio_to_file(output_file_name)
        i += 1


if __name__ == '__main__':
    main()
