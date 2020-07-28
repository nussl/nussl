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

# # High/Low Pass Filter
#

# +
import nussl
import matplotlib.pyplot as plt
import time

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

musdb = nussl.datasets.MUSDB18(
    download=True, sample_rate=16000,
    strict_sample_rate = False
)
i = 39
item = musdb[i]
mix = item['mix']
# -

separator = nussl.separation.benchmark.HighLowPassFilter(
    mix, 100)
estimates = separator()
estimates = {
    'Low': estimates[0],
    'High': estimates[1]
}
visualize_and_embed(estimates)

separator = nussl.separation.benchmark.HighLowPassFilter(
    mix, 400)
estimates = separator()
estimates = {
    'Low': estimates[0],
    'High': estimates[1]
}
visualize_and_embed(estimates)

end_time = time.time()
time_taken = end_time - start_time
print(f'Time taken: {time_taken:.4f} seconds')
