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

# # PROJET
#
# Fitzgerald, Derry, Antoine Liutkus, and Roland Badeau. 
# "Projection-based demixing of spatial audio." 
# IEEE/ACM Transactions on Audio, Speech, and Language 
# Processing 24.9 (2016): 1560-1572.
#
# Fitzgerald, Derry, Antoine Liutkus, and Roland Badeau. 
# "Projet—spatial audio separation using projections." 
# 2016 IEEE International Conference on Acoustics, 
# Speech and Signal Processing (ICASSP). IEEE, 2016.
#
#       @article{fitzgerald2016projection,
#         title={Projection-based demixing of spatial audio},
#         author={Fitzgerald, Derry and Liutkus, Antoine and Badeau, Roland},
#         journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
#         volume={24},
#         number={9},
#         pages={1560--1572},
#         year={2016},
#         publisher={IEEE}
#       }
#      @inproceedings{fitzgerald2016projet,
#        title={Projet—spatial audio separation using projections},
#        author={Fitzgerald, Derry and Liutkus, Antoine and Badeau, Roland},
#        booktitle={2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
#        pages={36--40},
#        year={2016},
#        organization={IEEE}
#       }


# +
import nussl
import matplotlib.pyplot as plt
import time
import numpy as np
import warnings
import torch

warnings.filterwarnings("ignore")
start_time = time.time()

nussl.utils.seed(0)

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
# -

# Setting up a signal for PROJET

# +
item = musdb[i]
sources = [
    item['sources']['drums'],
    item['sources']['other']
]

a = nussl.mixing.pan_audio_signal(sources[0], -35)
a_delays = [np.random.randint(1, 10) for _ in range(a.num_channels)]
a = nussl.mixing.delay_audio_signal(a, a_delays)

b = nussl.mixing.pan_audio_signal(sources[1], -15)
b_delays = [np.random.randint(1, 10) for _ in range(b.num_channels)]
b = nussl.mixing.delay_audio_signal(b, b_delays)

mix = a + b
# -

# Now running PROJET

# +
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
separator = nussl.separation.spatial.Projet(
    mix, num_sources=2, device=DEVICE, num_iterations=500)
estimates = separator()

estimates = {
    f'Source {i}': e for i, e in enumerate(estimates)
}

visualize_and_embed(estimates)
# -

end_time = time.time()
time_taken = end_time - start_time
print(f'Time taken: {time_taken:.4f} seconds')
