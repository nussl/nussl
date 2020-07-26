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

# # DUET
#
# Rickard, Scott. "The DUET blind source separation algorithm." 
# Blind speech separation. Springer, Dordrecht, 2007. 217-241.
#
#       @incollection{rickard2007duet,
#         title={The DUET blind source separation algorithm},
#         author={Rickard, Scott},
#         booktitle={Blind speech separation},
#         pages={217--241},
#         year={2007},
#         publisher={Springer}
#       }

# +
import nussl
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings("ignore")
start_time = time.time()

nussl.utils.seed(0)

def visualize_and_embed(sources):
    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    nussl.utils.visualize_sources_as_masks(sources,
        y_axis='linear', db_cutoff=-40, alpha_amount=2.0)
    plt.subplot(212)
    nussl.utils.visualize_sources_as_waveform(
        sources, show_legend=False)
    plt.show()
    nussl.play_utils.multitrack(sources)

audio_path = nussl.efz_utils.download_audio_file(
    'wsj_speech_mixture_ViCfBJj.mp3')
audio_signal = nussl.AudioSignal(audio_path)
separator = nussl.separation.spatial.Duet(
    audio_signal, num_sources=2)
estimates = separator()

estimates = {
    f'Speaker {i}': e for i, e in enumerate(estimates)
}

visualize_and_embed(estimates)
# -

end_time = time.time()
time_taken = end_time - start_time
print(f'Time taken: {time_taken:.4f} seconds')
