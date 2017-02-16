import os
import sys
import time

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)

import nussl


mixture = nussl.AudioSignal('../input/mixture/mixture.wav', duration = 30, offset = 60)
vocals = nussl.AudioSignal('../input/mixture/vocals.wav', duration = 30, offset = 60)
background = mixture - vocals

evaluation = nussl.Evaluation(ground_truth = [background, vocals],
                              compute_permutation = False,
                              ground_truth_labels = ['Vocals', 'Background'])


def evaluate(evaluation_object, sources, algorithm_name):
    print 'Evaluating %s' % algorithm_name
    evaluation_object.algorithm_name = algorithm_name
    evaluation_object.estimated_sources = sources

    start = time.time()
    evaluation_object.bss_eval_sources()
    evaluation_object.bss_eval_images()
    end = time.time()
    print end - start

ideal_mask = nussl.IdealMask(mixture, sources = [vocals])
ideal_mask.run()
sources = ideal_mask.make_audio_signals()[::-1]

evaluate(evaluation, sources, 'Ideal binary mask')

repet = nussl.Repet(mixture)
repet.run()
sources = repet.make_audio_signals()

evaluate(evaluation, sources, 'REPET')

repet_sim = nussl.RepetSim(mixture)
repet_sim.run()
sources = repet_sim.make_audio_signals()

evaluate(evaluation, sources, 'REPET-SIM')

ft2d = nussl.FT2D(mixture)
ft2d.run()
sources = ft2d.make_audio_signals()

evaluate(evaluation, sources, 'FT2D')

overlap_add = nussl.OverlapAdd(mixture, separation_method = 'REPET',
                               hop_size = 5, window_size = 10)
overlap_add.run()
sources = overlap_add.make_audio_signals()

evaluate(evaluation, sources, 'Overlap-Add: REPET')

overlap_add.separation_method = 'FT2D'
overlap_add.run()
sources = overlap_add.make_audio_signals()

evaluate(evaluation, sources, 'Overlap-Add: FT2D')

evaluation.write_scores_to_file('output/evaluation.json')