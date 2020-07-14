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

# # Independent Component Analysis
#
# Hyvärinen, Aapo, and Erkki Oja. 
# "Independent component analysis: algorithms and applications." 
# Neural networks 13.4-5 (2000): 411-430.
#
#
#     @article{hyvarinen2000independent,
#       title={Independent component analysis: algorithms and applications},
#       author={Hyv{\"a}rinen, Aapo and Oja, Erkki},
#       journal={Neural networks},
#       volume={13},
#       number={4-5},
#       pages={411--430},
#       year={2000},
#       publisher={Elsevier}
#     }

# +
import nussl
import matplotlib.pyplot as plt
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")
start_time = time.time()

def visualize_and_embed(sources):
    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    nussl.utils.visualize_sources_as_masks(sources,
        y_axis='mel', db_cutoff=-40, alpha_amount=2.0)
    plt.subplot(212)
    nussl.utils.visualize_sources_as_waveform(
        sources, show_legend=False)
    plt.show()
    nussl.play_utils.multitrack(sources)

musdb = nussl.datasets.MUSDB18(download=True)
i = 39
# -

# Setting up a signal that ICA can work on

# +
item = musdb[i]
sources = [item['sources']['other'], item['sources']['vocals']]

a = nussl.mixing.pan_audio_signal(sources[0], -35)
a_delays = [np.random.randint(1, 20) for _ in range(a.num_channels)]
a = nussl.mixing.delay_audio_signal(a, a_delays)

b = nussl.mixing.pan_audio_signal(sources[1], 35)
b_delays = [np.random.randint(1, 20) for _ in range(b.num_channels)]
b = nussl.mixing.delay_audio_signal(b, b_delays)

mix = a + b

audio_signals = [
    mix.make_audio_signal_from_channel(ch)
    for ch in range(mix.num_channels)
]
# -

# Now running ICA

# +
separator = nussl.separation.factorization.ICA(audio_signals)
estimates = separator()

estimates = {
    f'Source {i}': e for i, e in enumerate(estimates)
}

visualize_and_embed(estimates)
# -

end_time = time.time()
time_taken = end_time - start_time
print(f'Time taken: {time_taken:.4f} seconds')
