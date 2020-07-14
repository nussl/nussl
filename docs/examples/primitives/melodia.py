# -*- coding: utf-8 -*-
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

# # Melodia
#
# Salamon, Justin, and Emilia Gómez. 
# "Melody extraction from polyphonic music signals 
# using pitch contour characteristics." IEEE Transactions 
# on Audio, Speech, and Language Processing 20.6 (2012): 1759-1770.
#
#     @article{salamon2012melody,
#       title={Melody extraction from polyphonic music signals using pitch contour characteristics},
#       author={Salamon, Justin and G{\'o}mez, Emilia},
#       journal={IEEE Transactions on Audio, Speech, and Language Processing},
#       volume={20},
#       number={6},
#       pages={1759--1770},
#       year={2012},
#       publisher={IEEE}
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
    nussl.play_utils.multitrack(estimates, ['Background', 'Foreground', 'Synth Melody'])

audio_path = nussl.efz_utils.download_audio_file(
    'schoolboy_fascination_excerpt.wav')
audio_signal = nussl.AudioSignal(audio_path)
separator = nussl.separation.primitive.Melodia(
    audio_signal, mask_type='binary')
estimates = separator()
estimates.append(separator.melody_signal * .1)

visualize_and_embed(estimates)
# -

separator = nussl.separation.primitive.Melodia(
    audio_signal, mask_type='soft')
estimates = separator()
estimates.append(separator.melody_signal * .1)
visualize_and_embed(estimates)

separator = nussl.separation.primitive.Melodia(
    audio_signal, apply_vowel_filter=True, mask_type='binary')
estimates = separator()
estimates.append(separator.melody_signal * .1)
visualize_and_embed(estimates)

separator = nussl.separation.primitive.Melodia(
    audio_signal, apply_vowel_filter=True, 
    add_lower_octave=True, mask_type='binary')
estimates = separator()
estimates.append(separator.melody_signal * .1)
visualize_and_embed(estimates)

end_time = time.time()
time_taken = end_time - start_time
print(f'Time taken: {time_taken:.4f} seconds')
