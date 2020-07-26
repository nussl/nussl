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

# # Timbre clustering
#
# This method is similar to but not exactly like:
#
# Spiertz, Martin, and Volker Gnann. 
# "Source-filter based clustering for monaural blind source separation."
# Proceedings of the 12th International Conference on Digital Audio Effects. 2009.
#
#     @inproceedings{spiertz2009source,
#       title={Source-filter based clustering for monaural blind source separation},
#       author={Spiertz, Martin and Gnann, Volker},
#       booktitle={Proceedings of the 12th International Conference on Digital Audio Effects},
#       year={2009}
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

audio_path = nussl.efz_utils.download_audio_file(
    'marimba_timbre.mp3')
audio_signal = nussl.AudioSignal(audio_path)
separator = nussl.separation.primitive.TimbreClustering(
    audio_signal, 2, 50, mask_type='binary')
estimates = separator()

estimates = {
    f'Cluster {i}': e for i, e in enumerate(estimates)
}

visualize_and_embed(estimates)
# -

end_time = time.time()
time_taken = end_time - start_time
print(f'Time taken: {time_taken:.4f} seconds')
