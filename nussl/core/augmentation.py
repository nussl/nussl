import numpy as np
import ffmpeg
import librosa
import os
from .audio_signal import AudioSignal

"""

def augment(dataset: base_dataset, augment_proportion=1, num_augments=1, **kwargs):

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
    
    return augmented_dataset
"""

def time_stretch(signal, stretch_factor):
    """
    Linear Stretch on the time axis
    Args: 
        item: An item from a base_dataset
        factor_range: A tuple of length 2. Denotes start and end of possible ranges for factor. 
    Returns:
        augmented_item: A copy of the original item, with augmented sources. 
    """
    if not np.isscalar(stretch_factor) or stretch_factor <= 0:
        raise ValueError("stretch_factor must be a positve scalar")
    sample_rate = signal.sample_rate
    stretched_source = []
    audio_data = signal.audio_data

    for row in range(audio_data.shape[0]):
        audio_row = audio_data[row, :]
        if librosa.__version__ > "0.6.2":
            audio_row = np.asfortranarray(audio_row)
        stretched_source.append(librosa.effects.time_stretch(audio_row, stretch_factor))
    stretched_signal = AudioSignal(audio_data_array=np.array(stretched_source), sample_rate=sample_rate)

    return stretched_signal

def pitch_shift(signal, shift):
    """
    Pitch shift on the frequency axis
    Args: 
        signal: An Audio signal object
        shift: The number of half-steps to shift the audio. 
            Positive values increases the frequency of the signal
    Returns:
        augmented_item: A copy of the original item, with augmented sources. 
    """
    if not isinstance(shift, int):
        raise ValueError("The pitch shift must be an integer.")

    sample_rate = signal.sample_rate
    shifted_source = []
    audio_data = signal.audio_data

    for row in range(audio_data.shape[0]):
        audio_row = audio_data[row, :]
        if librosa.__version__ > "0.6.2":
            audio_row = np.asfortranarray(audio_row)
        shifted_source.append(librosa.effects.pitch_shift(audio_row, sample_rate, shift))
    shifted_signal = AudioSignal(audio_data_array=np.array(shifted_source), sample_rate=sample_rate)
    return shifted_signal

def loudness_scale(item, factor_range):
    """
    Loudness Scaling
    Args: 
        item: An item from a base_dataset
        factor_range: A tuple of length 2. Denotes start and end of possible ranges for factor. 
    Returns:
        augmented_item: A copy of the original item, with augmented sources. 
    """
    raise NotImplementedError

def low_pass(item, factor_range):
    """
    Applies low pass filter
    Args: 
        item: An item from a base_dataset
        factor_range: A tuple of length 2. Denotes start and end of possible ranges for low pass filter. 
    Returns:
        augmented_item: A copy of the original item, with augmented sources. 
    """
    raise NotImplementedError

def high_pass(item, factor_range):
    """
    Applies high pass filter
    Args: 
        item: An item from a base_dataset
        factor_range: A tuple of length 2. Denotes start and end of possible ranges for high pass filter. 
    Returns:
        augmented_item: A copy of the original item, with augmented sources. 
    """
    raise NotImplementedError

def tremolo(item, factor_range):
    """
    Applies tremolo filter
    Args: 
        item: An item from a base_dataset
        factor_range: A tuple of length 2, where each item is a tuple of length 2. 
            First tuple denotes range for modulation frequency, the second denotes range for modulation depth.
    Returns:
        augmented_item: A copy of the original item, with augmented sources. 
    """
    raise NotImplementedError

def vibrato(item, factor_range):
    """
    Applies vibrato filter
    Args: 
        item: An item from a base_dataset
        factor_range: A tuple of length 2, where each item is a tuple of length 2. 
            First tuple denotes range for modulation frequency, the second denotes range for modulation depth.
    Returns:
        augmented_item: A copy of the original item, with augmented sources. 
    """
    raise NotImplementedError

def igaussian_filter(item, factor_range):
    """
    Applies Inverse Gaussian filter
    Args: 
        item: An item from a base_dataset
        factor_range: A tuple of length 2, where each item is a tuple of length 2. 
            First tuple denotes range for frequency mean, the second denotes range for frequency standard deviation.
    Returns:
        augmented_item: A copy of the original item, with augmented sources. 
    """
    raise NotImplementedError
