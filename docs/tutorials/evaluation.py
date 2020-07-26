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

# Evaluating separation performance
# =================================
#
# In this notebook, we will demonstrate how one can use *nussl* to quickly and easily 
# compare different separation approaches. Here, we will evaluate the performance of 
# several simple vocal separation algorithms on a subset of the MUSDB18 dataset.
#
# First, let's load the dataset using *nussl*'s dataset utilities, and inspect an 
# item from the dataset using *nussl*'s plotting and playing utlities:

# +
import nussl
import numpy as np
import matplotlib.pyplot as plt
import json
import time

start_time = time.time()

# seed this notebook
nussl.utils.seed(0)

# this will download the 7 second clips from MUSDB
musdb = nussl.datasets.MUSDB18(download=True)
i = 40 #or get a random track like this: np.random.randint(len(musdb))

# helper for plotting and playing
def visualize_and_embed(sources):
    plt.figure(figsize=(10, 7))
    plt.subplot(211)
    nussl.utils.visualize_sources_as_masks(
        sources, db_cutoff=-60, y_axis='mel')
    plt.subplot(212)
    nussl.utils.visualize_sources_as_waveform(
        sources, show_legend=False)
    plt.tight_layout()
    plt.show()

    nussl.play_utils.multitrack(sources, ext='.wav')

item = musdb[i]
mix = item['mix']
sources = item['sources']

visualize_and_embed(sources)
# -

# So, there are four sources in each item of the MUSDB18 dataset: drums, bass, other, 
# and vocals. Since we're doing vocal separation, what we really care about is two 
# sources: vocals and accompaniment (drums + bass + other). So it'd be great if each 
# item in the dataset looked more like this:

# +
vocals = sources['vocals']
accompaniment = sources['drums'] + sources['bass'] + sources['other']

new_sources = {'vocals': vocals, 'accompaniment': accompaniment}
visualize_and_embed(new_sources)
# -

# When evaluating vocals separation, what we'll do is compare our estimate for 
# the vocals and the accompanient to the above ground truth isolated sources. 
# But first, there's a way in *nussl* to automatically group sources in a dataset 
# by type, using `nussl.datasets.transforms.SumSources`:

# +
tfm = nussl.datasets.transforms.SumSources([['drums', 'bass', 'other']])
# SumSources takes a list of lists, which each item in the list being 
# a group of sources that will be summed into a single source
musdb = nussl.datasets.MUSDB18(download=True, transform=tfm)

item = musdb[i]
mix = item['mix']
sources = item['sources']

visualize_and_embed(sources)
# -

# Now that we have a mixture and corresponding ground truth sources, let's 
# pump the mix through some of *nussl*'s separation algorithms and see what they sound like!

# REPET
# -----

repet = nussl.separation.primitive.Repet(mix)
repet_estimates = repet()
visualize_and_embed(repet_estimates)

# 2DFT
# ----

ft2d = nussl.separation.primitive.FT2D(mix)
ft2d_estimates = ft2d()
visualize_and_embed(ft2d_estimates)

# HPSS
# ----

hpss = nussl.separation.primitive.HPSS(mix)
hpss_estimates = hpss()[::-1]
# hpss gives harmonic then percussive
# so let's reverse the order of the list
visualize_and_embed(hpss_estimates)

# Putting it all together
# -----------------------
#
# Now that we have some estimates, let's evaluate the performance. 
# There are many ways to do this in *nussl*:
#
# 1. Original BSS Evaluation metrics:
#     - Source-to-distortion ratio (SDR): how well does the estimate 
#       match the ground truth source?
#     - Source-to-interference ratio (SIR): how well does the estimate 
#       suppress the other sources?
#     - Source-to-artifact ratio (SAR): how much musical/random noise 
#       is in the estimate?
#     - Source to Spatial Distortion Image (ISR): how well does the 
#       algorithm keep the source in the same spatial location?
# 2. New BSS Evaluation metrics: these metrics are refined versions 
#    of the originals and are argued to be more robust.
# 3. Precision and recall on binary masks: an older way to evaluate methods 
#    is to look at the values of the actual mask and the estimated mask and 
#    compute precision/recall over each time-frequency bin.
#    
# Let's extract each of these measures on the REPET estimates computed before.

# +
# make sources a list to feed into eval
sources_list = [sources['drums+bass+other'], sources['vocals']]

# 1. Original BSS Evaluation metrics
original_bss = nussl.evaluation.BSSEvalV4(
    sources_list, repet_estimates)
scores = original_bss.evaluate()

print(json.dumps(scores, indent=2))
# -

# The output dictionary of an evaluation method always looks like this: there is a 
# combination key, which indicates what combination of the estimates provided best
# matched to the sources, the permutation key, which can permute the estimates to
# match the sources (both of these are only computed when `compute_permutation = True`), 
# and dictionaries with each metric: SDR/SIR/ISR/SAR. Computing the other BSS Eval metrics 
# is just as easy:

new_bss = nussl.evaluation.BSSEvalScale(
    sources_list, repet_estimates)
scores = new_bss.evaluate()
print(json.dumps(scores, indent=2))

# To do the last, precision-recall one, we need ground truth binary masks to 
# compare to. First, let's convert the masks in our `repet` instance to be binary.

repet_binary_masks = [r.mask_to_binary(0.5) for r in repet.result_masks]

# Now, let's get the ideal binary mask using the `benchmark` methods in *nussl*:

ibm = nussl.separation.benchmark.IdealBinaryMask(mix, sources_list)
ibm_estimates = ibm()
visualize_and_embed(ibm_estimates)

# Now, we can evaluate the masks precision and recall:

prf = nussl.evaluation.PrecisionRecallFScore(
    ibm.result_masks, repet_binary_masks, 
    source_labels=['acc', 'vox'])
scores = prf.evaluate()
print(json.dumps(scores, indent=2))


# Great! But what do all of these numbers even mean? To establish the bounds 
# of performance of a separation algorithm, we need *upper* and *lower* baselines. 
# These numbers can be found by using the benchmark methods in *nussl*. 
# Let's get two lower baseline and an upper baseline. 
#
# **For the sake of brevity of output, let's look at the new BSSEval metrics.**
#
# We already have one upper baseline - the ideal binary mask. 
# How did that do?

# +
def _report_sdr(approach, scores):
    SDR = {}
    SIR = {}
    SAR = {}
    print(approach)
    print(''.join(['-' for i in range(len(approach))]))
    for key in scores:
        if key not in ['combination', 'permutation']:
            SDR[key] = np.mean(scores[key]['SI-SDR'])
            SIR[key] = np.mean(scores[key]['SI-SIR'])
            SAR[key] = np.mean(scores[key]['SI-SAR'])
            print(f'{key} SI-SDR: {SDR[key]:.2f} dB')
            print(f'{key} SI-SIR: {SIR[key]:.2f} dB')
            print(f'{key} SI-SAR: {SAR[key]:.2f} dB')
            print()
    print()

bss = nussl.evaluation.BSSEvalScale(
    sources_list, ibm_estimates,
    source_labels=['acc', 'vox'])
scores = bss.evaluate()
_report_sdr('Ideal Binary Mask', scores)
# -

# Let's get two lower baselines: using a simple high low pass filter, 
# and using the mixture as the estimate:

# +
mae = nussl.separation.benchmark.MixAsEstimate(
    mix, len(sources))
mae_estimates = mae()

bss = nussl.evaluation.BSSEvalScale(
    sources_list, mae_estimates,
    source_labels=['acc', 'vox'])
scores = bss.evaluate()
_report_sdr('Mixture as estimate', scores)

hlp = nussl.separation.benchmark.HighLowPassFilter(mix, 100)
hlp_estimates = hlp()

bss = nussl.evaluation.BSSEvalScale(
    sources_list, hlp_estimates,
    source_labels=['acc', 'vox'])
scores = bss.evaluate()
_report_sdr('High/low pass filter', scores)
# -

# Now that we've established upper and lower baselines, how did our methods do? 
# Let's write a function to run a separation algorithm, evaluate it, and 
# report its result on the mix.

# +
mae = nussl.separation.benchmark.MixAsEstimate(
    mix, len(sources))
hlp = nussl.separation.benchmark.HighLowPassFilter(
    mix, 100)
ibm = nussl.separation.benchmark.IdealBinaryMask(
    mix, sources_list)

hpss = nussl.separation.primitive.HPSS(mix)
ft2d = nussl.separation.primitive.FT2D(mix)
repet = nussl.separation.primitive.Repet(mix)


def run_and_evaluate(alg):
    alg_estimates = alg()
    
    if isinstance(alg, nussl.separation.primitive.HPSS):
        alg_estimates = alg_estimates[::-1]

    bss = nussl.evaluation.BSSEvalScale(
        sources_list, alg_estimates,
        source_labels=['acc', 'vox'])
    scores = bss.evaluate()
    _report_sdr(str(alg).split(' on')[0], scores)

for alg in [mae, hlp, hpss, repet, ft2d, ibm]:
    run_and_evaluate(alg)
# -

# We've now evaluated a bunch of algorithms on a single 7-second audio file. 
# Is this enough to say definitively one algorithm is better than others? 
# Probably not. When evaluating algorithms, one should always *listen* to 
# the separations as well as looking at metrics to report. One should also 
# make sure to compare against logical baselines, as well as do this on 
# challenging mixtures. 

end_time = time.time()
time_taken = end_time - start_time
print(f'Time taken: {time_taken:.4f} seconds')
