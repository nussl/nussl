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

# # HPSS
#
# Fitzgerald, Derry. “Harmonic/percussive separation using median filtering.” 
# 13th International Conference on Digital Audio Effects (DAFX10), Graz, 
# Austria, 2010.
#     
# Driedger, Müller, Disch. “Extending harmonic-percussive separation of audio.” 
# 15th International Society for Music Information Retrieval Conference 
# (ISMIR 2014) Taipei, Taiwan, 2014.
#
#     @inproceedings{fitzgerald2010harmonic,
#       title={Harmonic/percussive separation using median filtering},
#       author={Fitzgerald, Derry},
#       booktitle={Proc. of DAFX},
#       volume={10},
#       number={4},
#       year={2010}
#     }
#     
#     @inproceedings{driedger2014extending,
#       title={Extending Harmonic-Percussive Separation of Audio Signals.},
#       author={Driedger, Jonathan and M{\"u}ller, Meinard and Disch, Sascha},
#       booktitle={ISMIR},
#       pages={611--616},
#       year={2014}
#     }

# +
import nussl
import matplotlib.pyplot as plt
import time

start_time = time.time()

def visualize_and_embed(sources):
    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    nussl.utils.visualize_sources_as_masks(sources,
        y_axis='mel', db_cutoff=-60, alpha_amount=2.0)
    plt.subplot(212)
    nussl.utils.visualize_sources_as_waveform(
        sources, show_legend=False)
    plt.show()
    nussl.play_utils.multitrack(sources)

tfm = nussl.datasets.transforms.SumSources([
    ['vocals', 'other', 'bass']
])
musdb = nussl.datasets.MUSDB18(
    download=True, transform=tfm)
i = 40

item = musdb[i]
mix = item['mix']
sources = item['sources']

visualize_and_embed(sources)

# +
separator = nussl.separation.primitive.HPSS(
    mix, mask_type='binary')
estimates = separator()
estimates = {
    'Harmonic': estimates[0],
    'Percussive': estimates[1]
}

visualize_and_embed(estimates)
# -

end_time = time.time()
time_taken = end_time - start_time
print(f'Time taken: {time_taken:.4f} seconds')
