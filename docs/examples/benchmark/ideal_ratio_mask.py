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

# # Ideal Ratio Mask
#

# +
import nussl
import matplotlib.pyplot as plt
import time
import numpy as np

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
item = musdb[i]
mix = item['mix']
source_names = sorted(list(item['sources'].keys()))
sources = [item['sources'][k] for k in source_names]
# -

# Magnitude spectrum approximation

separator = nussl.separation.benchmark.IdealRatioMask(
    mix, sources, approach='msa')
estimates = separator()
estimates = {
    source_names[i]: e for i, e in enumerate(estimates)
}
visualize_and_embed(estimates)

# Truncated phase-sensitive spectrum approximation
#
# These are the default settings.

separator = nussl.separation.benchmark.IdealRatioMask(
    mix, sources, approach='psa', range_min=0.0, range_max=1.0)
estimates = separator()
estimates = {
    source_names[i]: e for i, e in enumerate(estimates)
}
visualize_and_embed(estimates)

# Phase-sensitive spectrum approximation
#
# Not truncating the phase works well for speech but seemingly not as
# much for music.

separator = nussl.separation.benchmark.IdealRatioMask(
    mix, sources, approach='psa', range_min=-np.inf, 
    range_max=np.inf)
estimates = separator()
estimates = {
    source_names[i]: e for i, e in enumerate(estimates)
}
visualize_and_embed(estimates)

end_time = time.time()
time_taken = end_time - start_time
print(f'Time taken: {time_taken:.4f} seconds')
