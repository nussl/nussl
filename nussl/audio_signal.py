#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import os.path
import numpy as np
import scipy.io.wavfile as wav
import librosa
import numbers
import audioread
import json
import warnings

import spectral_utils
import constants
import utils


class AudioSignal(object):
    """AudioSignal is the main entry point for the user or source separation algorithm to manipulate audio.

    The AudioSignal class is a container for all things related to audio data. It contains utilities for
    I/O, time-series and frequency domain manipulation, plotting, and much more. The AudioSignal class is used
    in all source separation objects in *nussl*.

    Parameters:
        path_to_input_file (str, optional): Path to an input file to open upon initialization. Audio gets loaded
            into ``self.audio_data``.
        audio_data_array (:obj:`np.ndarray`, optional): Numpy array containing a real-valued, time-series representation
            of the audio.
        offset (int, optional): Starting point of the section to be extracted in seconds.
            Defaults to 0
        duration (int, optional): Length of the signal to be extracted. Defaults to full length of the signal
        sample_rate (int, optional): sampling rate to read audio file at. Defaults to Constants.DEFAULT_SAMPLE_RATE
        stft (:obj:`np.ndarray`, optional): Optional pre-computed complex spectrogram data.
        stft_params (:obj:`StftParams`, optional):

    Examples:
        * create a new signal object:     ``signal = nussl.AudioSignal('sample_audio_file.wav')``
        * compute the spectrogram of the new signal object:   ``signal.stft()``
        * compute the inverse stft of a spectrogram:          ``sig.istft()``

    See Also:
        For a walk-through of AudioSignal features, see :ref:`audio_signal_basics` and :ref:`audio_signal_stft`.

    Attributes:
        audio_data (:obj:`np.ndarray`):
            Real-valued, uncompressed, time-domain representation of the audio.
            2D numpy array with shape `(n_channels, n_samples)`.
            ``None`` by default, this can be initialized at instantiation.
            Usually, this is expected to be floats. Some functions will convert to floats if not already.
        path_to_input_file (str): Path to the input file. ``None`` if this AudioSignal never loaded a file, i.e.,
            initialized with a np.array.
        sample_rate (int): Sample rate of this AudioSignal object.
        stft_data (:obj:`np.ndarray`): Complex-valued, frequency-domain representation of audio calculated by the
            Short-Time Fourier Transform (STFT).
            3D numpy array with shape `(n_frequency_bins, n_hops, n_channels)`.
            ``None`` by default, this can be initialized at instantiation.
        stft_params (:obj:`StftParams`): Container for all settings for doing a STFT. Has same lifespan as AudioSignal
            object.
  
    """

    def __init__(self, path_to_input_file=None, audio_data_array=None, stft=None,
                 sample_rate=constants.DEFAULT_SAMPLE_RATE, stft_params=None, offset=0, duration=None):

        self.path_to_input_file = path_to_input_file
        self._audio_data = None
        self.sample_rate = sample_rate
        self._active_start = None
        self._active_end = None

        # Assert that this object was only initialized in one way
        got_path = path_to_input_file is not None
        got_audio_array = audio_data_array is not None
        got_stft = stft is not None
        init_inputs = np.array([got_path, got_audio_array, got_stft])

        if len(init_inputs[init_inputs == True]) > 1:
            raise ValueError('Can only initialize AudioSignal object with one of [path, audio, stft]!')

        if path_to_input_file is not None:
            self.load_audio_from_file(self.path_to_input_file, offset, duration)
        elif audio_data_array is not None:
            self.load_audio_from_array(audio_data_array, sample_rate)

        # stft data
        self.stft_data = stft  # complex spectrogram data
        self.stft_params = spectral_utils.StftParams(self.sample_rate) if stft_params is None else stft_params
        self.use_librosa_stft = constants.USE_LIBROSA_STFT

    def __str__(self):
        return 'AudioSignal'

    ##################################################
    # Plotting
    ##################################################

    def plot_time_domain(self):
        """
        Not implemented yet -- will raise and exception
        Returns:

        """
        raise NotImplementedError('Not ready yet!')

    _NAME_STEM = 'audio_signal'

    def plot_spectrogram(self, file_name=None, ch=None):
        # TODO: use self.stft_data if not None
        # TODO: flatten to mono be default
        # TODO: make other parameters adjustable
        if file_name is None:
            name_stem = self.file_name if self.file_name is not None else self._NAME_STEM
        else:
            if os.path.isfile(file_name):
                name_stem = os.path.splitext(file_name)[0]
            else:
                name_stem = file_name

        if ch is None:
            if self.num_channels > 1:
                for i in range(1, self.num_channels+1):
                    name = name_stem + '_ch{}.png'.format(i)
                    spectral_utils.plot_stft(self.get_channel(i), name,
                                             sample_rate=self.sample_rate)
            else:
                name = name_stem + '.png'
                spectral_utils.plot_stft(self.get_channel(1), name,
                                         sample_rate=self.sample_rate)
        else:
            name = name_stem + '_ch{}.png'.format(ch)
            spectral_utils.plot_stft(self.get_channel(ch), name,
                                     sample_rate=self.sample_rate)

    ##################################################
    # Properties
    ##################################################

    # Constants for accessing _audio_data np.array indices
    _LEN = 1
    _CHAN = 0

    _STFT_BINS = 0
    _STFT_LEN = 1
    _STFT_CHAN = 2

    @property
    def signal_length(self):
        """ (int): Number of samples in the active region of `self.audio_data`
            The length of the audio signal represented by this object in samples
        """
        if self.audio_data is None:
            return None
        return self.audio_data.shape[self._LEN]

    @property
    def signal_duration(self):
        """ (float): Duration of audio in seconds.
            The length of the audio signal represented by this object in seconds
        """
        if self.signal_length is None:
            return None
        return self.signal_length / self.sample_rate

    @property
    def num_channels(self):
        """ (int): Number of channels this AudioSignal has.
            Defaults to returning number of channels in ``self.audio_data``. If that is ``None``, returns number of
            channels in ``self.stft_data``. If both are ``None`` then returns ``None``.
        """
        # TODO: what if about a mismatch between audio_data and stft_data
        if self.audio_data is not None:
            return self.audio_data.shape[self._CHAN]
        if self.stft_data is not None:
            return self.stft_data.shape[self._STFT_CHAN]
        return None

    @property
    def audio_data(self):
        """ (:obj:`np.ndarray`): Real-valued, uncompressed, time-domain representation of the audio.
            2D numpy array with shape `(n_channels, n_samples)`.
            ``None`` by default, this can be initialized at instantiation.
            Usually, this is expected to be floats. Some functions will convert to floats if not already.
        """
        if self._audio_data is None:
            return None

        start = 0
        end = self._audio_data.shape[self._LEN]

        if self._active_end is not None and self._active_end < end:
            end = self._active_end

        if self._active_start is not None and self._active_start > 0:
            start = self._active_start

        return self._audio_data[:, start:end]

    @audio_data.setter
    def audio_data(self, value):
        assert (type(value) == np.ndarray)

        self._audio_data = value

        if self._audio_data.ndim < 2:
            self._audio_data = np.expand_dims(self._audio_data, axis=self._CHAN)

    @property
    def file_name(self):
        """ (str): The name of the file wth extension, NOT the full path
        See Also:
            :ref: self.path_to_input_file
        """
        if self.path_to_input_file is not None:
            return os.path.split(self.path_to_input_file)[1]
        return None

    @property
    def time_vector(self):
        """ (:obj:`np.ndarray`): A 1D np array with timestamps (in seconds) for each sample in the time domain.
        """
        if self.signal_duration is None:
            return None
        return np.linspace(0.0, self.signal_duration, num=self.signal_length)

    @property
    def freq_vector(self):
        """ (:obj:`np.ndarray`): A 1D numpy array with frequency values that correspond
            to each frequency bin (vertical axis) for the STFT.
        Raises:
            AttributeError: If ``self.stft_data`` is ``None``. Run ``self.stft()`` before accessing this.
        """
        if self.stft_data is None:
            raise AttributeError('Cannot calculate freq_vector until self.stft() is run')
        return np.linspace(0.0, self.sample_rate // 2, num=self.stft_data.shape[self._STFT_BINS])

    @property
    def stft_length(self):
        """ (int): The number of time windows the STFT has.
        Raises:
            AttributeError: If ``self.stft_dat``a is ``None``. Run ``self.stft()`` before accessing this.
        """
        if self.stft_data is None:
            raise AttributeError('Cannot calculate stft_length until self.stft() is run')
        return self.stft_data.shape[self._STFT_LEN]

    @property
    def num_fft_bins(self):
        """ (int): Number of FFT bins in self.stft_data
        Raises:
            AttributeError: If ``self.stft_data`` is ``None``. Run ``self.stft()`` before accessing this.
        """
        if self.stft_data is None:
            raise AttributeError('Cannot calculate num_fft_bins until self.stft() is run')
        return self.stft_data.shape[self._STFT_BINS]

    @property
    def active_region_is_default(self):
        """ (bool): True if active region is the full length of ``self.audio_data``
        See Also:
            :ref: self.set_active_region
            :ref: self.set_active_region_to_default

        """
        return self._active_start == 0 and self._active_end == self._signal_length

    @property
    def _signal_length(self):
        """ (int): This is the length of the full signal, not just the active region.
        """
        if self._audio_data is None:
            return None
        return self._audio_data.shape[self._LEN]

    @property
    def power_spectrogram_data(self):
        """ (:obj:`np.ndarray`): Returns a real valued 2D ``np.array`` with power spectrogram data.
        The power spectrogram is defined as Re(STFT)^2, where ^2 is element-wise squaring
        of entries of the STFT. Same shape as ``self.stft_data``.
        Raises:
            AttributeError: if ``self.stft_data`` is ``None``. Run ``self.stft()`` before accessing this.
        """
        if self.stft_data is None:
            raise AttributeError('Cannot calculate power_spectrogram_data until self.stft() is run')
        return np.abs(self.stft_data) ** 2

    ##################################################
    # I/O
    ##################################################

    def load_audio_from_file(self, input_file_path, offset=0, duration=None):
        # type: (str, float, float) -> None
        """Loads an audio signal from a file

        Parameters:
            input_file_path (str): Path to input file.
            offset (float, optional): The starting point of the section to be extracted (seconds).
                Defaults to 0 seconds.
            duration (float, optional): Length of signal to load in second.
                signal_length of 0 means read the whole file. Defaults to the full length of the signal.

        """
        try:
            with audioread.audio_open(os.path.realpath(input_file_path)) as input_file:
                file_length = input_file.duration

            if offset > file_length:
                raise ValueError('offset is longer than signal!')

            if duration is not None and offset + duration >= file_length:
                warnings.warn('offset + duration are longer than the signal. Reading until end of signal...',
                              UserWarning)

            audio_input, self.sample_rate = librosa.load(input_file_path,
                                                         sr=None,
                                                         offset=offset,
                                                         duration=duration,
                                                         mono=False)

            # Change from fixed point to floating point
            if not np.issubdtype(audio_input.dtype, float):
                audio_input = audio_input.astype('float') / (np.iinfo(audio_input.dtype).max + 1.0)

            self.audio_data = audio_input

        except Exception as e:
            if isinstance(e, ValueError):  # This is the error we just raise, re-raise it
                raise e
            else:
                raise IOError("Cannot read from file, {file}".format(file=input_file_path))

        self.path_to_input_file = input_file_path
        self.set_active_region_to_default()

    def load_audio_from_array(self, signal, sample_rate=constants.DEFAULT_SAMPLE_RATE):
        """Loads an audio signal from a numpy array.

        Notes:
            Only accepts float arrays and int arrays of depth 16-bits.

        Parameters:
            signal (:obj:`np.ndarray`): Array containing the audio file signal sampled at sampleRate
            sample_rate (int, optional): the sample rate of signal. Default is Constants.DEFAULT_SAMPLE_RATE (44.1kHz)

        """
        assert (type(signal) == np.ndarray)

        self.path_to_input_file = None

        # Change from fixed point to floating point
        if not np.issubdtype(signal.dtype, float):
            if np.max(signal) > np.iinfo(np.dtype('int16')).max:
                raise ValueError('Please convert your array to 16-bit audio.')

            signal = signal.astype('float') / (np.iinfo(np.dtype('int16')).max + 1.0)

        self.audio_data = signal
        self.sample_rate = sample_rate
        self.set_active_region_to_default()

    def write_audio_to_file(self, output_file_path, sample_rate=None, verbose=False):
        """Outputs the audio signal to a file

        Parameters:
            output_file_path (str): Filename where output file will be saved.
            sample_rate (int, optional): The sample rate to write the file at. Default is ``self.sample_rate``.
            verbose (bool, optional): Print out a message if writing the file was successful.
        """
        if self.audio_data is None:
            raise Exception("Cannot write audio file because there is no audio data.")

        try:
            self.peak_normalize()

            if sample_rate is None:
                sample_rate = self.sample_rate

            audio_output = np.copy(self.audio_data)

            # TODO: better fix
            # convert to fixed point again
            if not np.issubdtype(audio_output.dtype, int):
                audio_output = np.multiply(audio_output, 2 ** (constants.DEFAULT_BIT_DEPTH - 1)).astype('int16')

            wav.write(output_file_path, sample_rate, audio_output.T)
        except Exception as e:
            print("Cannot write to file, {file}.".format(file=output_file_path))
            raise e
        if verbose:
            print("Successfully wrote {file}.".format(file=output_file_path))

    def set_active_region(self, start, end):
        """
        Determines the bounds of what gets returned when you access ``self.audio_data``.
        None of the data in ``self.audio_data`` is discarded when you set the active region, it merely becomes
        inaccessible until the active region is set back to default (i.e., the full length of the signal).

        This is useful for reusing a single ``AudioSignal`` object to do multiple operations on only select parts of the
        audio data.

        Warnings:
            Many functions will raise exceptions while the active region is not default. Be aware that adding,
            subtracting, concatenating, truncating, and other utilities may not be available.

        See Also:
            :ref: set_active_region_to_default
            :ref: active_region_is_default

        Args:
            start (int): Beginning of active region (in samples). Cannot be less than 0.
            end (int): End of active region (in samples). Cannot be larger than self.signal_length.

        """
        start, end = int(start), int(end)
        self._active_start = start if start >= 0 else 0
        self._active_end = end if end < self._signal_length else self._signal_length

    def set_active_region_to_default(self):
        """
        Resets the active region of this ``AudioSignal`` object to it default value of the entire ``audio_data`` array.

        """
        self._active_start = 0
        self._active_end = self._signal_length

    ##################################################
    #               STFT Utilities
    ##################################################

    def stft(self, window_length=None, hop_length=None, window_type=None, n_fft_bins=None, remove_reflection=True,
             overwrite=True, use_librosa=constants.USE_LIBROSA_STFT):
        """Computes the Short Time Fourier Transform (STFT) of ``self.audio_data``.

            The results of the STFT calculation can be accessed from ``self.stft_data``
            if ``self.stft_data`` is ``None`` prior to running this function or ``overwrite == True``

        Warning:
            If overwrite=True (default) this will overwrite any data in ``self.stft_data``!

        Args:
            window_length (int, optional): Amount of time (in samples) to do an FFT on
            hop_length (int, optional): Amount of time (in samples) to skip ahead for the new FFT
            window_type (str, optional): Type of scaling to apply to the window.
            n_fft_bins (int, optional): Number of FFT bins per each hop
            remove_reflection (bool, optional): Should remove reflection above Nyquist
            overwrite (bool, optional): Overwrite ``self.stft_data`` with current calculation
            use_librosa (bool, optional): Use *librosa's* stft function

        Returns:
            (:obj:`np.ndarray`) Calculated, complex-valued STFT from ``self.audio_data``, 3D numpy array
            with shape `(n_frequency_bins, n_hops, n_channels)`.

        """
        if self.audio_data is None or self.audio_data.size == 0:
            raise ValueError("No time domain signal (self.audio_data) to make STFT from!")

        window_length = self.stft_params.window_length if window_length is None else int(window_length)
        hop_length = self.stft_params.hop_length if hop_length is None else int(hop_length)
        window_type = self.stft_params.window_type if window_type is None else window_type
        n_fft_bins = self.stft_params.n_fft_bins if n_fft_bins is None else int(n_fft_bins)

        calculated_stft = self._do_stft(window_length, hop_length, window_type,
                                        n_fft_bins, remove_reflection, use_librosa)

        if overwrite:
            self.stft_data = calculated_stft

        return calculated_stft

    def _do_stft(self, window_length, hop_length, window_type, n_fft_bins, remove_reflection, use_librosa):
        if self.audio_data is None or self.audio_data.size == 0:
            raise ValueError('Cannot do stft without signal!')

        stfts = []

        stft_func = spectral_utils.librosa_stft_wrapper if use_librosa else spectral_utils.e_stft

        for i in range(1, self.num_channels + 1):
            stfts.append(stft_func(signal=self.get_channel(i), window_length=window_length,
                                   hop_length=hop_length, window_type=window_type,
                                   n_fft_bins=n_fft_bins, remove_reflection=remove_reflection))

        return np.array(stfts).transpose((1, 2, 0))

    def istft(self, window_length=None, hop_length=None, window_type=None, overwrite=True,
              reconstruct_reflection=False, use_librosa=constants.USE_LIBROSA_STFT):
        """Computes and returns the inverse Short Time Fourier Transform (iSTFT).

            The results of the iSTFT calculation can be accessed from ``self.audio_data``
            if ``self.audio_data`` is ``None`` prior to running this function or ``overwrite == True``

        Warning:
            If overwrite=True (default) this will overwrite any data in ``self.audio_data``!

        Args:
            window_length (int, optional): Amount of time (in samples) to do an FFT on
            hop_length (int, optional): Amount of time (in samples) to skip ahead for the new FFT
            window_type (str, optional): Type of scaling to apply to the window.
            reconstruct_reflection (bool, optional): Should reconstruct the reflection above Nyquist
            overwrite (bool, optional): Overwrite ``self.stft_data`` with current calculation
            use_librosa (bool, optional): Use *librosa's* stft function

        Returns:
            (:obj:`np.ndarray`) Calculated, real-valued iSTFT from ``self.stft_data``, 2D numpy array
            with shape `(n_channels, n_samples)`.

        """
        if self.stft_data is None or self.stft_data.size == 0:
            raise ValueError('Cannot do inverse STFT without self.stft_data!')

        window_length = self.stft_params.window_length if window_length is None else int(window_length)
        hop_length = self.stft_params.hop_length if hop_length is None else int(hop_length)
        # TODO: bubble up center
        window_type = self.stft_params.window_type if window_type is None else window_type

        calculated_signal = self._do_istft(window_length, hop_length, window_type,
                                           reconstruct_reflection, use_librosa)

        if overwrite or self.audio_data is None:
            self.audio_data = calculated_signal

        return calculated_signal

    def _do_istft(self, window_length, hop_length, window_type, reconstruct_reflection, use_librosa):
        if self.stft_data.size == 0:
            raise ValueError('Cannot do inverse STFT without self.stft_data!')

        signals = []

        istft_func = spectral_utils.librosa_istft_wrapper if use_librosa else spectral_utils.e_istft

        original_length = None if self.signal_length is None else self.signal_length

        for i in range(1, self.num_channels + 1):
            calculated_signal = istft_func(stft=self.get_stft_channel(i), window_length=window_length,
                                           hop_length=hop_length, window_type=window_type)

            if original_length is not None:
                calculated_signal = calculated_signal[:original_length]

            signals.append(calculated_signal)

        return np.array(signals)

    ##################################################
    #                  Utilities
    ##################################################

    def concat(self, other):
        """ Concatenate two ``AudioSignal`` objects (by concatenating ``self.audio_data``).

        Puts ``other.audio_data`` after ``self.audio_data``.

        Raises:
            AssertionError: If ``self.sample_rate != other.sample_rate``,
            ``self.num_channels != other.num_channels``, or ``self.active_region_is_default`` is ``False``.

        Parameters:
            (:obj:`AudioSignal`): ``AudioSignal`` to concatenate with the current one.
        """
        self._verify_audio(other, 'concat')

        self.audio_data = np.concatenate((self.audio_data, other.audio_data), axis=self._LEN)

    def truncate_samples(self, n_samples):
        """ Truncates the signal leaving only the first ``n_samples`` samples.
        This can only be done if ``self.active_region_is_default`` is True.

        Raises:
            Exception: If ``n_samples > self.signal_length`` or `self.active_region_is_default`` is ``False``.

        Args:
            n_samples: (int) number of samples that will be left.

        """
        if n_samples > self.signal_length:
            raise ValueError('n_samples must be less than self.signal_length!')

        if not self.active_region_is_default:
            raise Exception('Cannot truncate while active region is not set as default!')

        self.audio_data = self.audio_data[:, 0: n_samples]

    def truncate_seconds(self, n_seconds):
        """ Truncates the signal leaving only the first n_seconds.
        This can only be done if self.active_region_is_default is True.

        Raises:
            Exception: If ``n_seconds > self.signal_duration`` or `self.active_region_is_default`` is ``False``.

        Args:
            n_seconds: (float) number of seconds to truncate ``self.audio_data``.

        """
        if n_seconds > self.signal_duration:
            raise Exception('n_seconds must be shorter than self.signal_duration!')

        if not self.active_region_is_default:
            raise Exception('Cannot truncate while active region is not set as default!')

        n_samples = n_seconds * self.sample_rate
        self.truncate_samples(n_samples)

    def zero_pad(self, before, after):
        """ Adds zeros before and after the signal to all channels.
        Extends the length of self.audio_data by before + after.

        Raises:
            Exception: If `self.active_region_is_default`` is ``False``.

        Args:
            before: (int) number of zeros to be put before the current contents of self.audio_data
            after: (int) number of zeros to be put after the current contents fo self.audio_data

        """
        if not self.active_region_is_default:
            raise Exception('Cannot zero-pad while active region is not set as default!')

        for ch in range(1, self.num_channels + 1):
            self.audio_data = np.lib.pad(self.get_channel(ch), (before, after), 'constant', constant_values=(0, 0))

    def get_channel(self, n):
        """Gets the n-th channel from ``self.audio_data``. **1-based.**

        Raises:
            Exception: If not ``0 <= n < self.num_channels``.

        Parameters:
            n (int): index of channel to get. **1-based**
        Returns:
            (:obj:`np.array`): the audio data in the n-th channel of the signal, 1D
        """
        if n > self.num_channels:
            raise Exception(
                'Cannot get channel {0} when this object only has {1} channels!'.format(n, self.num_channels))

        if n <= 0:
            raise Exception('Cannot get channel {}. This will cause unexpected results'.format(n))

        return self._get_axis(self.audio_data, self._CHAN, n - 1)

    def get_stft_channel(self, n):
        """ Returns the n-th channel from ``self.stft_data``. **1-based.**

        Raises:
            Exception: If not ``0 <= n < self.num_channels``.

        Args:
            n: (int) index of stft channel to get. **1 based**

        Returns:
            (:obj:`np.array`): the stft data in the n-th channel of the signal, 1D
        """
        if n > self.num_channels:
            raise Exception(
                'Cannot get channel {0} when this object only has {1} channels!'.format(n, self.num_channels))

        if n <= 0:
            raise Exception('Cannot get channel {}. This will cause unexpected results'.format(n))

        return self._get_axis(self.stft_data, self._STFT_CHAN, n - 1)

    def get_power_spectrogram_channel(self, n):
        """ Returns the n-th channel from ``self.power_spectrogram_data``. **1-based.**

         Raises:
            Exception: If not ``0 <= n < self.num_channels``.

        Args:
            n: (int) index of power spectrogram channel to get. **1 based**

        Returns:
            (:obj:`np.array`): the power spectrogram data in the n-th channel of the signal, 1D
        """
        if n > self.num_channels:
            raise Exception(
                'Cannot get channel {0} when this object only has {1} channels!'.format(n, self.num_channels))

        # np.array helps with duck typing
        return self._get_axis(np.array(self.power_spectrogram_data), self._STFT_CHAN, n - 1)

    def peak_normalize(self, overwrite=True):
        """ Normalizes ``abs(self.audio_data)`` to 1.0.

            Warnings:
                If ``self.audio_data`` is not represented as floats this will convert the representation to floats!
        """
        max_val = 1.0
        max_signal = np.max(np.abs(self.audio_data))
        if max_signal > max_val:
            normalized = self.audio_data.astype('float') / max_signal
            if overwrite:
                self.audio_data = normalized
            return normalized

    def add(self, other):
        """Adds two audio signal objects.

        This does element-wise addition on the ``self.audio_data`` array.

        Raises:
            AssertionError: If ``self.sample_rate != other.sample_rate``,
            ``self.num_channels != other.num_channels``, or ``self.active_region_is_default`` is ``False``.

        Parameters:
            other (:obj:`AudioSignal`): Other ``AudioSignal`` to add.

        Returns:
            (:obj:`AudioSignal`): New ``AudioSignal`` object with the sum of ``self`` and ``other``.
        """
        return self + other

    def sub(self, other):
        """Subtracts two audio signal objects.

        This does element-wise subtraction on the ``self.audio_data`` array.

        Raises:
            AssertionError: If ``self.sample_rate != other.sample_rate``,
            ``self.num_channels != other.num_channels``, or ``self.active_region_is_default`` is ``False``.

        Parameters:
            other (:obj:`AudioSignal`): Other ``AudioSignal`` to subtract.

        Returns:
            (:obj:`AudioSignal`): New ``AudioSignal`` object with the difference between ``self`` and ``other``.
        """
        return self - other

    def audio_data_as_ints(self, bit_depth=constants.DEFAULT_BIT_DEPTH):
        """ Returns ``self.audio_data`` as a numpy array of signed ints with a specified bit-depth.

        Available bit-depths are: 8-, 16-, 24-, or 32-bits.

        Raises:
            TypeError: If ``bit_depth`` is not one of the above bit-depths.

        Notes:
            ``self.audio_data`` is regularly stored as an array of floats. This will not affect ``self.audio_data``.
        Args:
            bit_depth (int, optional): Bit depth of the integer array that will be returned.

        Returns:
            (:obj:`np.ndarray`): Integer representation of ``self.audio_data``.

        """
        if bit_depth not in [8, 16, 24, 32]:
            raise TypeError('Cannot convert self.audio_data to integer array of bit depth = {}'.format(bit_depth))

        int_type = 'int' + str(bit_depth)

        return np.multiply(self.audio_data, 2 ** (constants.DEFAULT_BIT_DEPTH - 1)).astype(int_type)

    def to_json(self):
        """ Converts this ``AudioSignal`` object to JSON.

        See Also:
            :ref: self.from_json

        Returns:
            (str): JSON representation of the current ``AudioSignal`` object.

        """
        return json.dumps(self, default=AudioSignal._to_json_helper)

    @staticmethod
    def _to_json_helper(o):
        if not isinstance(o, AudioSignal):
            raise TypeError
        import copy
        d = copy.copy(o.__dict__)
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = utils.json_ready_numpy_array(v)
        d['__class__'] = o.__class__.__name__
        d['__module__'] = o.__module__
        d['stft_params'] = o.stft_params.to_json()
        return d

    @staticmethod
    def from_json(json_string):
        """ Creates a new ``AudioSignal`` object from a JSON encoded ``AudioSignal`` string.

        For best results, ``json_string`` should be created from ``AudioSignal.to_json()``.

        See Also:
            :ref: self.to_json

        Args:
            json_string (string): a json encoded ``AudioSignal`` string

        Returns:
            (obj:`AudioSignal`): an ``AudioSignal`` object based on the parameters in JSON string

        """
        return json.loads(json_string, object_hook=AudioSignal._from_json_helper)

    @staticmethod
    def _from_json_helper(json_dict):
        if '__class__' in json_dict:
            class_name = json_dict.pop('__class__')
            module = json_dict.pop('__module__')
            if class_name != AudioSignal.__name__ or module != AudioSignal.__module__:
                raise TypeError
            a = AudioSignal()
            stft_params = json_dict.pop('stft_params')
            a.stft_params = spectral_utils.StftParams.from_json(stft_params)
            for k, v in json_dict.items():
                if isinstance(v, dict) and constants.NUMPY_JSON_KEY in v:
                    a.__dict__[k] = utils.json_numpy_obj_hook(v[constants.NUMPY_JSON_KEY])
                else:
                    a.__dict__[k] = v if not isinstance(v, unicode) else v.encode('ascii')
            return a
        else:
            return json_dict

    def rms(self):
        """ Calculates the root-mean-square of ``self.audio_data``.
        Returns:
            (float): Root-mean-square of ``self.audio_data``.

        """
        return np.sqrt(np.mean(np.square(self.audio_data)))

    def to_mono(self, overwrite=False):
        """ Converts ``self.audio_data`` to mono by averaging every sample.

        Warning:
            If overwrite=True (default) this will overwrite any data in ``self.audio_data``!

        Args:
            overwrite (bool, optional): If ``True`` this function will overwrite ``self.audio_data``.

        Returns:
            (:obj:`np.array`): Mono-ed version of ``self.audio_data``.

        """
        mono = np.mean(self.audio_data, axis=self._CHAN)
        if overwrite:
            self.audio_data = mono
        return mono

    ##################################################
    #              Operator overloading
    ##################################################

    def __add__(self, other):
        self._verify_audio(other, 'add')

        if self.signal_length > other.signal_length:
            combined = np.copy(self.audio_data)
            combined[:, :other.signal_length] += other.audio_data
        else:
            combined = np.copy(other.audio_data)
            combined[:, :self.signal_length] += self.audio_data

        return AudioSignal(audio_data_array=combined)

    def __sub__(self, other):
        self._verify_audio(other, 'subtract')

        if self.signal_length > other.signal_length:
            combined = np.copy(self.audio_data)
            combined[:, :other.signal_length] -= other.audio_data
        else:
            combined = np.copy(other.audio_data)
            combined[:, :self.signal_length] -= self.audio_data

        return AudioSignal(audio_data_array=combined)

    def _verify_audio(self, other, op):
        if self.num_channels != other.num_channels:
            raise Exception('Cannot ' + op + ' with two signals that have a different number of channels!')

        if self.sample_rate != other.sample_rate:
            raise Exception('Cannot' + op + 'with two signals that have different sample rates!')

        if not self.active_region_is_default:
            raise Exception('Cannot' + op + 'while active region is not set as default!')

    def __iadd__(self, other):
        return self + other

    def __isub__(self, other):
        return self - other

    def __mul__(self, other):
        assert isinstance(other, numbers.Real)
        raise NotImplemented('Not implemented yet.')

    def __len__(self):
        return self.signal_length

    def __eq__(self, other):
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                if not np.array_equal(v, other.__dict__[k]):
                    return False
            elif v != other.__dict__[k]:
                return False
        return True

    def __ne__(self, other):
        return not self == other

    ##################################################
    #              Private utils
    ##################################################

    @staticmethod
    def _get_axis(array, axis_num, i):
        if array.ndim == 2:
            if axis_num == 0:
                return array[i, :]
            elif axis_num == 1:
                return array[:, i]
            else:
                return None
        elif array.ndim == 3:
            if axis_num == 0:
                return array[i, :, :]
            elif axis_num == 1:
                return array[:, i, :]
            elif axis_num == 2:
                return array[:, :, i]
            else:
                return None
        else:
            return None
