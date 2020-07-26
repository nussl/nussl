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

# # REPET
#
# Rafii, Zafar, and Bryan Pardo. 
# "Repeating pattern extraction technique (REPET): 
# A simple method for music/voice separation." IEEE transactions 
# on audio, speech, and language processing 21.1 (2012): 73-84.
#
#     @article{rafii2012repeating,
#       title={Repeating pattern extraction technique (REPET): A simple method for music/voice separation},
#       author={Rafii, Zafar and Pardo, Bryan},
#       journal={IEEE transactions on audio, speech, and language processing},
#       volume={21},
#       number={1},
#       pages={73--84},
#       year={2012},
#       publisher={IEEE}
#     }

# +
import nussl
import matplotlib.pyplot as plt
import time

start_time = time.time()

audio_path = nussl.efz_utils.download_audio_file(
    'historyrepeating_7olLrex.wav')
audio_signal = nussl.AudioSignal(audio_path)
separator = nussl.separation.primitive.Repet(
    audio_signal, mask_type='binary')
estimates = separator()

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
# -

end_time = time.time()
time_taken = end_time - start_time
print(f'Time taken: {time_taken:.4f} seconds')
