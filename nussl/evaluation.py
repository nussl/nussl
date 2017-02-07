from audio_signal import AudioSignal
from mir_eval.separation import *
import numpy as np

class Evaluation(object):
    """Lets you load ground truth AudioSignals and estimated AudioSignals and compute sepation 
    evaluation criteria (SDR, SIR, SAR and delta SDR, delta SIR, delta SAR).

    Parameters:
        ground_truth: ground truth audio sources that make up the mixture. Consists of a list of 
        AudioSignal objects that sum up to the mixture. This must be provided.
        estimated_sources: Estimated audio sources. These sum up to the mixture and don't have to 
        be in the same order as ground_truth. This can be provided later and swapped dynamically
        to compare different audio source separation approaches.
        ground_truth_labels: Labels for the sources in ground truth. Used to interpret the 
        results later.
        sample_rate: Sample rate for all ground truth and estimated sources. Defaults to 
        the sample rate of the first AudioSignal in ground_truth.
        do_mono: whether to do evaluation using mono sources to multichannel sources.
        compute_permutation: True if you can't guarantee that ground_truth and estimated_sources
        are in the same order. False if you can. It'll be a bit faster if False. 
        Defaults to True.
        segment_size: when computing evaluation metrics, you can do them by segment instead 
        of for the whole track. Segment size defines how long each segment is. 
        Defaults to 30 seconds.
        hop_size: when computing evaluation metrics, you can do them by segment instead of 
        for the whole track. Hop size defines how much to hop between segments. 
        Defaults to 15 seconds.
    Examples:
  
    """
    def __init__(self, ground_truth=None, estimated_sources=None, ground_truth_labels=None, sample_rate=None, do_mono=None, compute_permutation=None, hop_size=None, segment_size=None):
        self.ground_truth = ground_truth
        if self.ground_truth is None:
            raise Exception('Cannot initialize Evaluation object without ground truth!')
        self.ground_truth_labels = ground_truth_labels
        if self.ground_truth_labels is None:
            self.grouth_labels = ['Source %d' % i for i in range(len(ground_truth))]
        self.estimated_sources = estimated_sources
        self.sample_rate = sample_rate
        if self.sample_rate is None:
            self.sample_rate = ground_truth[0].sample_rate
        
        self.compute_permutation = True if compute_permutation is None else compute_permutation
        self.do_mono = False if do_mono is True else do_mono
        if do_mono:
            self.num_channels = 1
        else:
            self.num_channels = ground_truth[0].num_channels
        self.segment_size = 30 if segment_size is None else segment_size
        self.hop_size = 15 if hop_size is None else hop_size
    
    def validate():
        if self.estimated_sources = None:
            raise Exception('Must set estimated_sources to run evaluation!')
        estimated_lengths = [x.signal_length for x in estimated_sources]
        reference_lengths = [x.signal_length for x in ground_truth]
        if len(set(estimated_lengths)) > 1:
            raise Exception('All AudioSignals in estimated_sources must be the same length!')
        if len(set(reference_lengths)) > 1:
            raise Exception('All AudioSignals in ground_truth must be the same length!')
    
    def to_mono():
        self.validate()
        for i, audio in enumerate(self.ground_truth):
            mono = audio.to_mono()
            self.ground_truth[i] = AudioSignal(audio_data_array = mono, sample_rate = self.sample_rate)
        for i, audio in enumerate(self.estimated_sources):
            mono = audio.to_mono()
            self.estimated_Sources[i] = AudioSignal(audio_data_array = mono, sample_rate = self.sample_rate)
        

    def transform_sources_to_array():
        estimated_source_array = np.stack([x.audio_data for x in estimated_sources], axis = -1)
        reference_source_array = np.stack([x.audio_data for x in ground_truth], axis = -1)
        return reference_source_array, estimated_source_array

    def bss_eval_sources():
        self.validate()

    def bss_eval_images():
        self.validate()

    def bss_eval_sources_framewise():
        self.validate()

    def bss_eval_images_framewise():
        self.validate()
