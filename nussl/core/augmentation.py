from ..datasets import base_dataset
from . import AudioSignal
import numpy as np
import ffmpeg
import librosa
import os
from ..datasets import transforms


def augment(dataset: base_dataset, augment_proportion=1, num_augments=1, **kwargs):
    """
    The augment function will take a dataset object that implements base_dataset,
    and with augment a proportion of the the datasets any of the following augmentations:

    Time stretching: Linearly scale the time axis in respect to a central portion. 
    Pitch shifting: Increase or decrease the sounds of the AudioSignal by a number of half steps
    Remixing: Change the loudness of each source independently.
    Loudness Scaling: Change the loudness of all sources by the same magnitude.
    Inverse gaussian filtering: Multiply each tf bin by the factor
    (1 - Gaussian(f)), where f is frequency, and Gaussian's parameters 
    are mean frequency, and standard deviation.
    Tremolo: Apply a Tremolo Effect to all sources and mixtures.
    Vibrato: Apply a Vibrato Effect to all sources and mixtures.

    Scale factors and parameters to the gaussian are all uniformly randomly
    chosen from a predetermined range.

    Args:
        dataset: A dataset object where each item is a dictionary containing "mix" and 
            "sources". This object should implement nussl.datasets.base_dataset
        augment_proportion: Indicates the proportion of the dataset to be augmented.
        num_augments: Number of augmented versions to make from each song. 

        The remaining arguments handle the ranges of possible values for augmentations. These are kwargs. 
        The arugment name is the name of the augmentation, and the value is the parameters for that specific augmentation.

        The following are tuples of length 2.
        time_stretch: Indicates range of factors for the time stretch, where (min_stretch, max_stretch)
        pitch_shift: Indicates range of shifts by number of half steps for the pitch shift, where (min_shift, max_shift)
        remix: Indicates range of factors for independent loudness scaling premixing, where (min_factor, max_factor)
        loudness_scale: Indicates range of factors for loudness scalings, where (min_factor, max_factor)
        low_pass: Indicates range of thresholds for low pass filters where (min_threshold, max_thresold)
        high_pass: Indicates range of thresholds for high pass filters where (min_threshold, max_threshold)

        The following are tuples of length 2, where each element is also a tuple of length 2. 
        tremolo: Indicates ranges for parameters to tremolo function, where ((min_modulation_frequency, max_modulation_frequency)
            ,(min_modulation_depth, max_modulation_depth))
        vibrato: Indicates ranges for parameters to vibrato function, where ((min_modulation_frequency, max_modulation_frequency)
            ,(min_modulation_depth, max_modulation_depth))
        igaussian_filter: Indicates ranges for parameters to the Inverse Gaussian, where ((min_mean, max_mean), 
            (min_standard_deviation, max_standard_derivations))

    Returns: 
        augmented_dataset: List of augmented mix-sources dictionaries. 
    """

    if augment_proportion < 0 or augment_proportion > 1:
        raise ValueError("augment_proportion must be between 0 and 1 inclusive.")
    if num_augments < 1 and not isinstance(num_augments, (int, np.int)):
        raise ValueError("num_augments should be positive integer.")
    
    proportion_dataset = np.random.choice(dataset, int(augment_proportion * len(dataset)), replace=False)
    augmented_dataset = []

    if not kwargs:
        raise RuntimeWarning("No Augmentations were passed to augment(). ")
        return augmented_dataset

    # Preform augmentations
    for _ in range(num_augments):
        for item in proportion_dataset:
            augmented_item = None
            for function, ranges in kwargs.items():
                ## This use of exec will allow us to easily support more augmentations
                augmented_item = eval(f"transforms._{function}(item, ranges)")
            augmented_dataset.append(augmented_item)
    
    print(augmented_dataset)
    return augmented_dataset



