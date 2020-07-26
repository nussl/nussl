# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 2DFT
#
# Seetharaman, Prem, Fatemeh Pishdadian, and Bryan Pardo.
# "Music/voice separation using the 2d fourier transform." 2017 
# IEEE Workshop on Applications of Signal Processing to Audio and 
# Acoustics (WASPAA). IEEE, 2017.
#
#     @inproceedings{seetharaman2017music,
#       title={Music/voice separation using the 2d fourier transform},
#       author={Seetharaman, Prem and Pishdadian, Fatemeh and Pardo, Bryan},
#       booktitle={2017 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
#       pages={36--40},
#       year={2017},
#       organization={IEEE}
#     }
#

# +
import nussl
import matplotlib.pyplot as plt
import time

start_time = time.time()

def visualize_and_embed(estimates):
    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    nussl.utils.visualize_sources_as_masks({
        'Background': estimates[0], 'Foreground': estimates[1]}, 
        y_axis='mel', db_cutoff=-60, alpha_amount=2.0)

    plt.subplot(212)
    nussl.utils.visualize_sources_as_waveform({
        'Background': estimates[0], 'Foreground': estimates[1]},
        show_legend=False)
    plt.show()
    nussl.play_utils.multitrack(estimates, ['Background', 'Foreground'])

audio_path = nussl.efz_utils.download_audio_file(
    'schoolboy_fascination_excerpt.wav')
audio_signal = nussl.AudioSignal(audio_path)
ft2d = nussl.separation.primitive.FT2D(
    audio_signal, mask_type='binary')
estimates = ft2d()

visualize_and_embed(estimates)
# -
ft2d = nussl.separation.primitive.FT2D(
    audio_signal, mask_type='soft')
estimates = ft2d()
visualize_and_embed(estimates)

ft2d = nussl.separation.primitive.FT2D(
    audio_signal, mask_type='binary', use_bg_2dft=False)
estimates = ft2d()
visualize_and_embed(estimates)

end_time = time.time()
time_taken = end_time - start_time
print(f'Time taken: {time_taken:.4f} seconds')
