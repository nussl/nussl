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

# # Robust Principal Component Analysis
#
# Huang, Po-Sen, et al. "Singing-voice separation from monaural 
# recordings using robust principal component analysis." 
# 2012 IEEE International Conference on Acoustics, Speech 
# and Signal Processing (ICASSP). IEEE, 2012.
#
#
#     @inproceedings{huang2012singing,
#       title={Singing-voice separation from monaural recordings using robust principal component analysis},
#       author={Huang, Po-Sen and Chen, Scott Deeann and Smaragdis, Paris and Hasegawa-Johnson, Mark},
#       booktitle={2012 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
#       pages={57--60},
#       year={2012},
#       organization={IEEE}
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

audio_path = nussl.efz_utils.download_audio_file(
    'schoolboy_fascination_excerpt.wav')
audio_signal = nussl.AudioSignal(audio_path)
          
separator = nussl.separation.factorization.RPCA(audio_signal)
estimates = separator()

estimates = {
    'Low-rank source': estimates[0],
    'Sparse source': estimates[1]
}

visualize_and_embed(estimates)
# -

end_time = time.time()
time_taken = end_time - start_time
print(f'Time taken: {time_taken:.4f} seconds')
