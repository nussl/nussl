"""
Provides utilities for running nussl algorithms that do not belong to
any specific algorithm or that are shared between algorithms.
"""

import warnings

import numpy as np
import torch
import random
from .. import musdb
import librosa
from . import constants

import os
from contextlib import contextmanager

def seed(random_seed, set_cudnn=False):
    """
    Seeds all random states in nussl with the same random seed
    for reproducibility. Seeds ``numpy``, ``random`` and ``torch``
    random generators. 

    For full reproducibility, two further options must be set
    according to the torch documentation:

    https://pytorch.org/docs/stable/notes/randomness.html

    To do this, ``set_cudnn`` must be True. It defaults to 
    False, since setting it to True results in a performance
    hit.

    Args:
        random_seed (int): integer corresponding to random seed to 
        use.
        set_cudnn (bool): Whether or not to set cudnn into determinstic
        mode and off of benchmark mode. Defaults to False.
    """

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    if set_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def find_peak_indices(input_array, n_peaks, min_dist=None, do_min=False, threshold=0.5):
    """
    This function will find the indices of the peaks of an input n-dimensional numpy array.
    This can be configured to find max or min peak indices, distance between the peaks, and
    a lower bound, at which the algorithm will stop searching for peaks (or upper bound if
    searching for max). Used exactly the same as :func:`find_peak_values`.

    This function currently only accepts 1-D and 2-D numpy arrays.

    Notes:
        * This function only returns the indices of peaks. If you want to find peak values,
        use :func:`find_peak_values`.

        * min_dist can be an int or a tuple of length 2.
            If input_array is 1-D, min_dist must be an integer.
            If input_array is 2-D, min_dist can be an integer, in which case the minimum
            distance in both dimensions will be equal. min_dist can also be a tuple if
            you want each dimension to have a different minimum distance between peaks.
            In that case, the 0th value in the tuple represents the first dimension, and
            the 1st value represents the second dimension in the numpy array.

    Args:
        input_array: a 1- or 2- dimensional numpy array that will be inspected.
        n_peaks: (int) maximum number of peaks to find
        min_dist: (int) minimum distance between peaks. Default value: len(input_array) / 4
        do_min: (bool) if True, finds indices at minimum value instead of maximum
        threshold: (float) the value (scaled between 0.0 and 1.0)

    Returns:
        peak_indices: (list) list of the indices of the peak values

    """
    input_array = np.array(input_array, dtype=float)

    if input_array.ndim > 2:
        raise ValueError('Cannot find peak indices on data greater than 2 dimensions!')

    is_1d = input_array.ndim == 1
    zero_dist = zero_dist0 = zero_dist1 = None
    min_dist = len(input_array) // 4 if min_dist is None else min_dist

    if is_1d:
        zero_dist = min_dist
    else:
        if type(min_dist) is int:
            zero_dist0 = zero_dist1 = min_dist
        elif len(min_dist) == 1:
            zero_dist0 = zero_dist1 = min_dist[0]
        else:
            zero_dist0, zero_dist1 = min_dist

    # scale input_array between [0.0, 1.0]
    input_array -= np.min(input_array)
    input_array /= np.max(input_array)

    # flip sign if doing min
    input_array = -input_array if do_min else input_array

    # throw out everything below threshold
    input_array = np.multiply(input_array, (input_array >= threshold))

    # check to make sure we didn't throw everything out
    if np.size(np.nonzero(input_array)) == 0:
        raise ValueError('Threshold set incorrectly. No peaks above threshold.')
    if np.size(np.nonzero(input_array)) < n_peaks:
        warnings.warn('Threshold set such that there will be less peaks than n_peaks.')

    peak_indices = []
    for i in range(n_peaks):
        # np.unravel_index for 2D indices e.g., index 5 in a 3x3 array should be (1, 2)
        # Also, wrap in list for duck typing
        cur_peak_idx = list(np.unravel_index(
            np.argmax(input_array.flatten()), input_array.shape
        ))

        # zero out peak and its surroundings
        if is_1d:
            cur_peak_idx = cur_peak_idx[0]
            peak_indices.append(cur_peak_idx)
            lower, upper = _set_array_zero_indices(cur_peak_idx, zero_dist, len(input_array))
            input_array[lower:upper] = 0
        else:
            peak_indices.append(cur_peak_idx)
            lower0, upper0 = _set_array_zero_indices(cur_peak_idx[0], zero_dist0,
                                                     input_array.shape[0])
            lower1, upper1 = _set_array_zero_indices(cur_peak_idx[1], zero_dist1,
                                                     input_array.shape[1])
            input_array[lower0:upper0, lower1:upper1] = 0

        if np.sum(input_array) == 0.0:
            break

    return peak_indices


def _set_array_zero_indices(index, zero_distance, max_len):
    lower = index - zero_distance
    upper = index + zero_distance + 1
    lower = 0 if lower < 0 else lower
    upper = max_len if upper >= max_len else upper
    return int(lower), int(upper)


def complex_randn(shape):
    """
    Returns a complex-valued numpy array of random values with shape :param:`shape`.

    Args:
        shape (tuple): Tuple of ints that will be the shape of the resultant complex numpy array.

    Returns:
        (:obj:`np.ndarray`): a complex-valued numpy array of random values with shape `shape`
    """
    return np.random.randn(*shape) + 1j * np.random.randn(*shape)


def _get_axis(array, axis_num, i):
    """
    Will get index 'i' along axis 'axis_num' using np.take.

    Args:
        array (:obj:`np.ndarray`): Array to fetch axis of.
        axis_num (int): Axes to retrieve.
        i (int): Index to retrieve.

    Returns:
        The value at index :param:`i` along axis :param:`axis_num`
    """

    return np.take(array, i, axis_num)


def _slice_along_dim(data, dim, start, end):
    """
    Takes a slice of data along a dim between a start and an end. Agnostic to
    whether the data is a numpy array or a torch tensor.
    
    Args:
        data (np.ndarray or torch.Tensor): Data to slice.
        dim (int): Dimension along which to do the slicing.
        start (int): Start of the slice.
        end (int): End of the slice
    """
    if dim > 3:
        raise ValueError("Unsupported for dim > 4")
    if dim >= len(data.shape):
        raise ValueError(f"dim {dim} too high for data.shape {data.shape}!")

    if dim == 0:
        return data[start:end, ...]
    elif dim == 1:
        return data[:, start:end, ...]
    elif dim == 2:
        return data[:, :, start:end, ...]
    elif dim == 3:
        return data[:, :, :, start:end, ...]


def _format(string):
    """ Formats a class name correctly for checking function and class names.
        Strips all non-alphanumeric chars and makes lowercase.
    """
    return ''.join(list(filter(str.isalnum, string))).lower()


def musdb_track_to_audio_signals(track):
    """
    Converts a musdb track to a dictionary of AudioSignal objects.
    
    Args:
        track (musdb.audio_classes.MultiTrack): MultiTrasack object 
            containing stems that will each be turned into AudioSignal
            objects.

    Returns:
        (2-tuple): tuple containing the mixture AudioSignal and a dictionary of
            the sources.
    """
    # lazy load to prevent circular imports
    from .audio_signal import AudioSignal

    mixture = AudioSignal(audio_data_array=track.audio, sample_rate=track.rate)
    mixture.path_to_input_file = track.name
    stems = track.stems
    sources = {}

    for k, v in sorted(track.sources.items(), key=lambda x: x[1].stem_id):
        sources[k] = AudioSignal(
            audio_data_array=stems[v.stem_id],
            sample_rate=track.rate
        )
        sources[k].path_to_input_file = f'musdb/{track.name}_{k}.wav'

    return mixture, sources


def audio_signals_to_musdb_track(mixture, sources_dict, targets_dict):
    """
    Converts :class:`AudioSignal` objects to ``musdb`` :class:`Track` objects that
    contain the mixture, the ground truth sources, and the targets for use with the ``mus_eval``
    implementation of BSS-Eval and ``musdb``.

    See Also:
        * More information on ``musdb``:  `Github<https://github.com/sigsep/sigsep-mus-db>`
            and `documentation<http://musdb.readthedocs.io/>`
        * More information on ``mus_eval``: `Github<https://github.com/sigsep/sigsep-mus-eval>`
            and `documentation<https://sigsep.github.io/sigsep-mus-eval/>`
        * :class:`BSSEvalV4` for *nussl*'s interface to BSS-Eval v4.

    Examples:
        .. code-block:: python
            :linenos:
            import nussl
            signal = nussl.AudioSignal(nussl.efz_utils.download_audio_file('HistoryRepeating.wav'))

            repet = nussl.Repet(signal)
            repet.run()

            bg, fg = repet.make_audio_signals()

            src_dict = {'vocals': fg, 'accompaniment': bg}
            target = nussl.core.constants.STEM_TARGET_DICT
            track = nussl.utils.audio_signals_to_musdb_track(signal, src_dict, target)

    Args:
        mixture (:class:`AudioSignal`): The :class:`AudioSignal` object that contains the mixture.
        sources_dict (dict): Dictionary where the keys are the labels for the sources and values
            are the associated :class:`AudioSignal` objects.
        targets_dict (dict): Dictionary where the keys are the labels for the sources (as above)
            and the values are weights.

    Returns:
        (:obj:`musdb.MultiTrack`) populated as specified by inputs.
    """
    verify_audio_signal_list_strict(list(sources_dict.values()) + [mixture])

    path = mixture.path_to_input_file if mixture.path_to_input_file else "None"
    fname = mixture.file_name if mixture.file_name else "None"
    track = musdb.audio_classes.MultiTrack(path=path, name=fname, is_wav=True)
    track.audio = mixture.audio_data.T
    track.rate = mixture.sample_rate

    stems = [track.audio]

    for name, target_srcs in list(targets_dict.items()):
        if name in sources_dict:
            stems.append(sources_dict[name].audio_data.T)

    track._stems = np.array(stems)
    return track


def verify_audio_signal_list_lax(audio_signal_list):
    """
    Verifies that an input (:param:`audio_signal_list`) is a list of :ref:`AudioSignal` objects.
    If not so, attempts to correct the list (if possible) and returns the corrected list.

    Args:
        audio_signal_list (list): List of :ref:`AudioSignal` objects

    Returns:
        audio_signal_list (list): Verified list of :ref:`AudioSignal` objects.

    """
    # Lazy load to prevent a circular reference upon initialization
    from .audio_signal import AudioSignal

    if isinstance(audio_signal_list, AudioSignal):
        audio_signal_list = [audio_signal_list]
    elif isinstance(audio_signal_list, list):
        if not all(isinstance(s, AudioSignal) for s in audio_signal_list):
            raise ValueError('All input objects must be AudioSignal objects!')
        if not all(s.has_data for s in audio_signal_list):
            raise ValueError('All AudioSignal objects in input list must have data!')
    else:
        raise ValueError(
            'audio_signal_list must be a list of or a single AudioSignal objects!')

    return audio_signal_list


def verify_audio_signal_list_strict(audio_signal_list):
    """
    Verifies that an input (:param:`audio_signal_list`) is a list of :ref:`AudioSignal` objects and
    that they all have the same sample rate and same number of channels. If not true,
    attempts to correct the list (if possible) and returns the corrected list.

    Args:
        audio_signal_list (list): List of :ref:`AudioSignal` objects

    Returns:
        audio_signal_list (list): Verified list of :ref:`AudioSignal` objects, that all have
        the same sample rate and number of channels.

    """
    audio_signal_list = verify_audio_signal_list_lax(audio_signal_list)

    if not all(audio_signal_list[0].sample_rate == s.sample_rate for s in audio_signal_list):
        raise ValueError('All input AudioSignal objects must have the same sample rate!')

    if not all(audio_signal_list[0].num_channels == s.num_channels for s in audio_signal_list):
        raise ValueError('All input AudioSignal objects must have the same number of channels!')

    if not all(audio_signal_list[0].signal_length == s.signal_length for s in audio_signal_list):
        raise ValueError('All input AudioSignal objects must have the same signal length!')

    return audio_signal_list


def visualize_gradient_flow(named_parameters, n_bins=50):
    """
    Visualize the gradient flow through the named parameters of a PyTorch model. 

    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as 
    "visualize_gradient_flow(self.model.named_parameters())" to visualize 
    the gradient flow
    
    Args:
        named_parameters (generator): Generator object yielding name and parameters
          for each layer in a PyTorch model.
        n_bins (int): Number of bins to use for each histogram. Defaults to 50.
    """
    import matplotlib.pyplot as plt

    data = []

    for n, p in named_parameters:
        if p.requires_grad and "bias" not in n:
            if p.grad is not None:
                _data = p.grad.cpu().data.numpy().flatten()
                lower = np.percentile(_data, 10)
                upper = np.percentile(_data, 90)
                _data = _data[_data >= lower]
                _data = _data[_data <= upper]
                n = n.split('layers.')[-1]
                data.append((n, _data, np.abs(_data).mean()))

    _data = [d[1] for d in sorted(data, key=lambda x: x[-1])]
    _names = [d[0] for d in sorted(data, key=lambda x: x[-1])]

    plt.hist(_data, len(_data) * n_bins, histtype='step', fill=False,
             stacked=True, label=_names)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2)


def visualize_spectrogram(audio_signal, ch=0, do_mono=False, x_axis='time',
                          y_axis='linear', **kwargs):
    """
    Wrapper around `librosa.display.specshow` for usage with AudioSignals.

    Args:
        audio_signal (AudioSignal): AudioSignal to plot
        ch (int, optional): Which channel to plot. Defaults to 0.
        do_mono (bool, optional): Make the AudioSignal mono. Defaults to False.
        x_axis (str, optional): x_axis argument to librosa.display.specshow. Defaults to 'time'.
        y_axis (str, optional): y_axis argument to librosa.display.specshow. Defaults to 'linear'.
        kwargs: Additional keyword arguments to librosa.display.specshow.
    """
    import librosa.display

    if do_mono:
        audio_signal = audio_signal.to_mono(overwrite=False)

    if y_axis == 'mel':
        # Monkey patch for https://github.com/librosa/librosa/issues/1240
        data = librosa.feature.melspectrogram(audio_signal.get_channel(ch),
                                              sr=audio_signal.sample_rate)
        kwargs.update({'fmax': audio_signal.sample_rate / 2.})
    else:
        data = np.abs(audio_signal.stft())[..., ch]

    data = librosa.amplitude_to_db(data, ref=np.max)
    librosa.display.specshow(data, x_axis=x_axis, y_axis=y_axis,
                             sr=audio_signal.sample_rate,
                             hop_length=audio_signal.stft_params.hop_length,
                             **kwargs)


def visualize_waveform(audio_signal, ch=0, do_mono=False, x_axis='time', **kwargs):
    """
    Wrapper around `librosa.display.waveplot` for usage with AudioSignals.
    
    Args:
        audio_signal (AudioSignal): AudioSignal to plot
        ch (int, optional): Which channel to plot. Defaults to 0.
        do_mono (bool, optional): Make the AudioSignal mono. Defaults to False.
        x_axis (str, optional): x_axis argument to librosa.display.waveplot. Defaults to 'time'.
        kwargs: Additional keyword arguments to librosa.display.waveplot.
    """
    import librosa.display
    import matplotlib.pyplot as plt

    if do_mono:
        audio_signal = audio_signal.to_mono(overwrite=False)
    
    data = np.asfortranarray(audio_signal.audio_data[ch])
    librosa.display.waveplot(data, sr=audio_signal.sample_rate, x_axis=x_axis, **kwargs)
    plt.ylabel('Amplitude')

def visualize_sources_as_waveform(audio_signals, ch=0, do_mono=False, x_axis='time', 
                                  colors=None, alphas=None, show_legend=True, **kwargs):
    """
    Visualizes a dictionary or list of sources with overlapping waveforms with transparency.
    
    The labels of each source are either the key, if a dictionary, or the 
    path to the input audio file, if a list.
    
    Args:
        audio_signals (list or dict): List or dictionary of audio signal objects to be
          plotted.
        ch (int, optional): Which channel to plot. Defaults to 0.
        do_mono (bool, optional): Make each AudioSignal mono. Defaults to False.
        x_axis (str, optional): x_axis argument to librosa.display.waveplot. Defaults to 'time'.
        colors (list, optional): Sequence of colors to use for each signal. 
          Defaults to None, which uses the default matplotlib color cycle.
        alphas (list, optional): Sequence of alpha transparency to use for each signal. 
          Defaults to None.
        kwargs: Additional keyword arguments to librosa.display.waveplot.
    """
    import matplotlib.pyplot as plt

    if isinstance(audio_signals, list):
        audio_signals = {
            f'{i}:{a.path_to_input_file}': a 
            for i, a in enumerate(audio_signals)
        }

    sorted_keys = sorted(
        audio_signals.keys(),
        key=lambda k: audio_signals[k].rms().mean(),
        reverse=True
    )

    alphas = (
        np.linspace(0.25, .75, len(audio_signals)) 
        if alphas is None else alphas
    )
    colors = (
        plt.rcParams['axes.prop_cycle'].by_key()['color'] 
        if colors is None else colors
    )

    for i, key in enumerate(sorted_keys):
        val = audio_signals[key]
        color = colors[i % len(audio_signals)]
        visualize_waveform(val, ch=ch, do_mono=do_mono, x_axis=x_axis, 
                           alpha=alphas[i % len(audio_signals)],
                           label=key, color=color)

    if show_legend:
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2)

def visualize_sources_as_masks(audio_signals, ch=0, do_mono=False, x_axis='time', 
                               y_axis='linear', db_cutoff=-60, colors=None, alphas=None, 
                               alpha_amount=1.0, show_legend=True, **kwargs):
    """
    Visualizes a dictionary or list of sources with overlapping waveforms with transparency.
    
    The labels of each source are either the key, if a dictionary, or the 
    path to the input audio file, if a list.
    
    Args:
        audio_signals (list or dict): List or dictionary of audio signal objects to be
          plotted.
        ch (int, optional): Which channel to plot. Defaults to 0.
        do_mono (bool, optional): Make each AudioSignal mono. Defaults to False.
        x_axis (str, optional): x_axis argument to librosa.display.waveplot. Defaults to 'time'.
        colors (list, optional): Sequence of colors to use for each signal. 
          Defaults to None, which uses the default matplotlib color cycle.
        alphas (list, optional): Sequence of alpha transparency to use for each signal. 
          Defaults to None.
        kwargs: Additional keyword arguments to librosa.display.specshow.
    """
    import matplotlib.pyplot as plt
    import matplotlib
    import librosa.display
    from .. import datasets

    if isinstance(audio_signals, list):
        audio_signals = {
            f'{i}:{a.path_to_input_file}': a 
            for i, a in enumerate(audio_signals)
        }

    if do_mono:
        for key in audio_signals:
            audio_signals[key] = audio_signals[key].to_mono()

    sorted_keys = sorted(
        audio_signals.keys(),
        key=lambda k: audio_signals[k].rms().mean(),
        reverse=True
    )

    source_names = sorted(list(audio_signals.keys()))
    mix = sum(audio_signals.values())
    data = {
        'mix': mix,
        'sources': audio_signals
    }
    data = datasets.transforms.PhaseSensitiveSpectrumApproximation()(data)

    colors = (
        plt.rcParams['axes.prop_cycle'].by_key()['color'] 
        if colors is None else colors
    )

    # construct each image with alpha values
    masks = data['source_magnitudes'] / (np.maximum(
            data['mix_magnitude'][..., None], data['source_magnitudes']) 
            + constants.EPSILON
        )
    legend_elements = []

    silence_mask = librosa.amplitude_to_db(np.abs(mix.stft()), ref=np.max) > db_cutoff
    masks *= silence_mask[..., None]

    y_coords = librosa.display.__mesh_coords(y_axis, None, masks.shape[0], 
        sr=mix.sample_rate, hop_length=mix.stft_params.hop_length)
    x_coords = librosa.display.__mesh_coords(x_axis, None, masks.shape[1],
        sr=mix.sample_rate, hop_length=mix.stft_params.hop_length)

    extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]

    for j, key in enumerate(sorted_keys):
        i = source_names.index(key)
        mask = masks[..., ch, i]
        color = colors[j % len(colors)]

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            'custom', ['white', color])
        image = cmap(mask)
        image[:, :, -1] = mask ** alpha_amount
        plt.imshow(image, origin='lower', aspect='auto', 
            interpolation='none', extent=extent)

        legend_elements.append(
            matplotlib.patches.Patch(facecolor=color, label=key))

    axes = librosa.display.__check_axes(None)

    axes.set_xlim(x_coords.min(), x_coords.max())
    axes.set_ylim(y_coords.min(), y_coords.max())

    # Set up axis scaling
    librosa.display.__scale_axes(axes, x_axis, 'x')
    librosa.display.__scale_axes(axes, y_axis, 'y')

    # Construct tickers and locators
    librosa.display.__decorate_axis(axes.xaxis, x_axis)
    librosa.display.__decorate_axis(axes.yaxis, y_axis)

    
    if show_legend:
        plt.legend(handles=legend_elements,  
            bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2)

@contextmanager
def _close_temp_files(tmpfiles):
    """
    Utility function for creating a context and closing all temporary files
    once the context is exited. For correct functionality, all temporary file
    handles created inside the context must be appended to the ```tmpfiles```
    list.

    This function is taken wholesale from Scaper.

    Args:
        tmpfiles (list): List of temporary file handles
    """
    def _close():
        for t in tmpfiles:
            try:
                t.close()
                os.unlink(t.name)
            except:
                pass
    try:
        yield
    except:
        _close()
        raise
    _close()
