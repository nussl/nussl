#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for PrecisionRecallFScore class

"""
import unittest
import os
import numpy as np
import sklearn.metrics

import nussl


class PrecisionRecallFScoreUnitTest(unittest.TestCase):

    def test_simple(self):
        drums_path = os.path.join('..', 'Input', 'src1.wav')
        flute_path = os.path.join('..', 'Input', 'src2.wav')

        drums = nussl.AudioSignal(drums_path)
        flute = nussl.AudioSignal(flute_path)
        flute.truncate_samples(drums.signal_length)

        mixture = drums + flute

        repet = nussl.Repet(mixture, mask_type=nussl.separation.BinaryMask)
        repet_mask_list = repet()

        ideal_mask = nussl.IdealMask(mixture, [drums, flute], mask_type=nussl.separation.BinaryMask)
        ideal_mask_list = ideal_mask()

        prf = nussl.PrecisionRecallFScore(ideal_mask_list, repet_mask_list)
        prf_scores = prf.evaluate()

    def test_prf_values(self):
        sizes = [128, 256, 512, 1024, 2048]
        for size in sizes:
            mask1_array = np.random.randint(0, 2, size=[size, size])
            mask1 = nussl.separation.BinaryMask(mask1_array)

            mask2_array = np.random.randint(0, 2, size=[size, size])
            mask2 = nussl.separation.BinaryMask(mask2_array)

            prf = nussl.PrecisionRecallFScore([mask1], [mask2])
            prf_scores = prf.evaluate()

            precision = sklearn.metrics.precision_score(mask1_array.ravel(), mask2_array.ravel())
            recall = sklearn.metrics.recall_score(mask1_array.ravel(), mask2_array.ravel())
            f1_score = sklearn.metrics.f1_score(mask1_array.ravel(), mask2_array.ravel())
            accuracy = sklearn.metrics.accuracy_score(mask1_array.ravel(), mask2_array.ravel())

            assert prf_scores['Source 0']['Precision'] == precision
            assert prf_scores['Source 0']['Recall'] == recall
            assert prf_scores['Source 0']['F1-Score'] == f1_score
            assert prf_scores['Source 0']['Accuracy'] == accuracy
