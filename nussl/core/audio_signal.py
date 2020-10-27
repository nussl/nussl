import copy
import numbers
import os.path
import warnings
from collections import namedtuple

import audioread
import librosa
import numpy as np
import scipy.io.wavfile as wav
import scipy
from scipy.signal import check_COLA
import soundfile as sf
import pyloudnorm

from . import constants
from . import utils
from . import masks
from . import effects

__all__ = ['AudioSignal', 'STFTParams', 'AudioSignalException']

STFTParams = namedtuple('STFTParams',
                        ['window_length', 'hop_length', 'window_type']
                        )
STFTParams.__new__.__defaults__ = (None,) * len(STFTParams._fields)
"""
STFTParams object is a container that holds STFT parameters - window_length, 
hop_length, and window_type. Not all parameters need to be specified. Ones that
are not specified will be inferred by the AudioSignal parameters and the settings
in `nussl.core.constants`.
"""


class AudioSignal(object):
    """

    **Overview**

    :class:`AudioSignal` is the main entry and exit point for all source separation algorithms
    in *nussl*. The :class:`AudioSignal` class is a general container for all things related to
    audio data. It contains utilities for:

    * Input and output from an array or from a file,
    * Time-series and frequency domain manipulation,
    * Plotting and visualizing,
    * Playing audio within a terminal or jupyter notebook,
    * Applying a mask to estimate signals

    and more. The :class:`AudioSignal` class is used in all source separation objects in *nussl*.

    :class:`AudioSignal` object stores time-series audio data as a 2D ``numpy`` array in
    :attr:`audio_data` (see :attr:`audio_data` for details) and stores Short-Time Fourier Transform
    data as 3D ``numpy`` array in :ref:`stft_data` (see :attr:`stft_data` for details).


    **Initialization**

    There are a few options for initializing an :class:`AudioSignal` object. The first is to
    initialize an empty :class:`AudioSignal` object, with no parameters:

     >>> import nussl
     >>> signal = nussl.AudioSignal()

    In this case, there is no data stored in :attr:`audio_data` or in :attr:`stft_data`, though
    these attributes can be updated at any time after the object has been created.

    Additionally, an :class:`AudioSignal` object can be loaded with exactly one of the following:

        1. A path to an input audio file (see :func:`load_audio_from_file` for details).
        2. A `numpy` array of 1D or 2D real-valued time-series audio data.
        3. A `numpy` array of 2D or 3D complex-valued time-frequency STFT data.

    :class:`AudioSignal` will throw an error if it is initialized with more than one of the
    previous at once.

    Here are examples of all three of these cases:

     .. code-block:: python
        :linenos:

        import numpy as np
        import nussl

        # Initializing an empty AudioSignal object:
        sig_empty = nussl.AudioSignal()

        # Initializing from a path:
        file_path = 'my/awesome/mixture.wav'
        sig_path = nussl.AudioSignal(file_path)

        # Initializing with a 1D or 2D numpy array containing audio data:
        aud_1d = np.sin(np.linspace(0.0, 1.0, 48000))
        sig_1d = nussl.AudioSignal(audio_data_array=aud_1d, sample_rate=48000)

        # FYI: The shape doesn't matter, nussl will correct for it
        aud_2d = np.array([aud_1d, -2 * aud_1d])
        sig_2d = nussl.AudioSignal(audio_data_array=aud_2d)

        # Initializing with a 2D or 3D numpy array containing STFT data:
        stft_2d = np.random.rand((513, 300)) + 1j * np.random.rand((513, 300))
        sig_stft_2d = nussl.AudioSignal(stft=stft_2d)

        # Two channels of STFT data:
        stft_3d = nussl.utils.complex_randn((513, 300, 2))
        sig_stft_3d = nussl.AudioSignal(stft=stft_3d)

        # Initializing with more than one of the above methods will raise an exception:
        sig_exception = nussl.AudioSignal(audio_data_array=aud_2d, stft=stft_2d)

    When initializing from a path, :class:`AudioSignal` can read many types of audio files,
    provided that your computer has the backends installed to understand the corresponding codecs.
    *nussl* uses ``librosa``'s `load` function to read in audio data. See librosa's documentation
    for details: https://github.com/librosa/librosa#audioread

    Once initialized with a single type of data (time-series or time-frequency), there are methods
    to compute an STFT from time-series data (:func:`stft`) and vice versa (:func:`istft`).

    **Sample Rate**

    The sample rate of an :class:`AudioSignal` object is set upon initialization. If initializing
    from a path, the sample rate of the :class:`AudioSignal` object inherits the native sample
    rate from the file. If initialized with an audio or stft data array, the sample rate is passed
    in as an optional argument. In these cases, with no sample rate explicitly defined, the default
    sample rate is 44.1 kHz (CD quality). If this argument is provided when reading from a file
    and the provided sample rate does not match the native sample rate of the file,
    :class:`AudioSignal` will resample the data from the file so that it matches the provided
    sample rate.

    Notes:
        There is no guarantee that data in :attr:`audio_data` corresponds to data in
        :attr:`stft_data`. E.g., when an :class:`AudioSignal` object is initialized with
        :attr:`audio_data` of an audio mixture, its :attr:`stft_data` is ``None`` until :func:`stft`
        is called. Once :func:`stft` is called and a mask is applied to :attr:`stft_data` (via some
        algorithm), the :attr:`audio_data` in this :class:`AudioSignal` object still contains data
        from the original mixture that it was initialized with even though :attr:`stft_data`
        contains altered data. (To hear the results, simply call :func:`istft` on the
        :class:`AudioSignal` object.) It is up to the user to keep track of the contents of
        :attr:`audio_data` and :attr:`stft_data`.

    See Also:
        For a walk-through of AudioSignal features, see :ref:`audio_signal_basics` and
        :ref:`audio_signal_stft`.

    Arguments:
        path_to_input_file (``str``): Path to an input file to load upon initialization. Audio
            gets loaded into :attr:`audio_data`.
        audio_data_array (:obj:`np.ndarray`): 1D or 2D numpy array containing a real-valued,
            time-series representation of the audio.
        stft (:obj:`np.ndarray`): 2D or 3D numpy array containing pre-computed complex-valued STFT
            data.
        label (``str``): A label for this :class:`AudioSignal` object.
        offset (``float``): Starting point of the section to be extracted (in seconds) if
            initializing from  a file.
        duration (``float``): Length of the signal to read from the file (in seconds). Defaults to
            full length of the signal (i.e., ``None``).
        sample_rate (``int``): Sampling rate of this :class:`AudioSignal` object.

    Attributes:
        path_to_input_file (``str``): Path to the input file. ``None`` if this AudioSignal never
            loaded a file, i.e., initialized with a ``np.ndarray``.
        label (``str``): A user-definable label for this :class:`AudioSignal` object.
        applied_effects (``list`` of ``effects.FilterFunction``): Effects applied to this 
        :class:`AudioSignal` object. For more information, see apply_effects. 
        effects_chain (``list`` of ``effects.FilterFunction``): Effects queues to be applied to 
        this :class:`AudioSignal` object. For more information, see apply_effects. 
  
    """

    def __init__(self, path_to_input_file=None, audio_data_array=None, stft=None, label=None,
                 sample_rate=None, stft_params=None, offset=0, duration=None):

        self.path_to_input_file = path_to_input_file
        self._audio_data = None
        self.original_signal_length = None
        self._stft_data = None
        self._sample_rate = None
        self._active_start = None
        self._active_end = None
        self.label = label

        # Assert that this object was only initialized in one way
        got_path = path_to_input_file is not None
        got_audio_array = audio_data_array is not None
        got_stft = stft is not None
        init_inputs = np.array([got_path, got_audio_array, got_stft])

        # noinspection PyPep8
        if len(init_inputs[init_inputs == True]) > 1:  # ignore inspection for clarity
            raise AudioSignalException('Can only initialize AudioSignal object with one and only '
                                       'one of {path, audio, stft}!')

        if path_to_input_file is not None:
            self.load_audio_from_file(self.path_to_input_file, offset, duration, sample_rate)
        elif audio_data_array is not None:
            self.load_audio_from_array(audio_data_array, sample_rate)

        if self._sample_rate is None:
            self._sample_rate = constants.DEFAULT_SAMPLE_RATE \
                if sample_rate is None else sample_rate

        self.stft_data = stft  # complex spectrogram data
        self.stft_params = stft_params

        # Effects
        self._effects_chain = []
        self._effects_applied = []

    def __str__(self):
        dur = f'{self.signal_duration:0.3f}' if self.signal_duration else '[unknown]'
        return (
            f"{self.__class__.__name__} "
            f"({self.label if self.label else 'unlabeled'}): "
            f"{dur} sec @ "
            f"{self.path_to_input_file if self.path_to_input_file else 'path unknown'}, "
            f"{self.sample_rate if self.sample_rate else '[unknown]'} Hz, "
            f"{self.num_channels if self.num_channels else '[unknown]'} ch."
        )

    ##################################################
    #                 Properties
    ##################################################

    @property
    def signal_length(self):
        """
        ``int``
            Number of samples in the active region of :attr:`audio_data`.
            The length of the audio signal represented by this object in samples.

        See Also:
            * :func:`signal_duration` for the signal duration in seconds.
            * :func:`set_active_region_to_default` for information about active regions.
        """
        if self.audio_data is None:
            return self.original_signal_length
        return self.audio_data.shape[constants.LEN_INDEX]

    @property
    def signal_duration(self):
        """
        ``float``
            Duration of the active region of :attr:`audio_data` in seconds.
            The length of the audio signal represented by this object in seconds.

        See Also:
            * :func:`signal_length` for the signal length in samples.
            * :func:`set_active_region_to_default` for information about active regions.
        """
        if self.signal_length is None:
            return None
        return self.signal_length / self.sample_rate

    @property
    def num_channels(self):
        """
        ``int``
            Number of channels this :class:`AudioSignal` has.
            Defaults to returning number of channels in :attr:`audio_data`. If that is ``None``,
            returns number of channels in :attr:`stft_data`. If both are ``None`` then returns
            ``None``.

        See Also:
            * :func:`is_mono`
            * :func:`is_stereo`
        """
        # TODO: what about a mismatch between audio_data and stft_data??
        if self.audio_data is not None:
            return self.audio_data.shape[constants.CHAN_INDEX]
        if self.stft_data is not None:
            return self.stft_data.shape[constants.STFT_CHAN_INDEX]
        return None

    @property
    def is_mono(self):
        """
        ``bool``
            Whether or not this signal is mono (i.e., has exactly **one** channel). First
            looks at :attr:`audio_data`, then (if that's ``None``) looks at :attr:`stft_data`.

        See Also:
            * :func:`num_channels`
            * :func:`is_stereo`
        """
        return self.num_channels == 1

    @property
    def is_stereo(self):
        """
        ``bool``
            Whether or not this signal is stereo (i.e., has exactly **two** channels). First
            looks at :attr:`audio_data`, then (if that's ``None``) looks at :attr:`stft_data`.

        See Also:
            * :func:`num_channels`
            * :func:`is_mono`
        """
        return self.num_channels == 2

    @property
    def audio_data(self):
        """
        ``np.ndarray``
            Stored as a ``numpy`` :obj:`np.ndarray`, :attr:`audio_data` houses the raw, uncompressed
            time-domain audio data in the :class:`AudioSignal`. Audio data is stored with shape
            ``(n_channels, n_samples)`` as an array of floats.

            ``None`` by default, can be initialized upon object instantiation or set at any time by
            accessing this attribute or calling :func:`load_audio_from_array`. It is recommended to
            set :attr:`audio_data` by using :func:`load_audio_from_array` if this
            :class:`AudioSignal` has been initialized without any audio or STFT data.

        Raises:
            :class:`AudioSignalException`
                If set incorrectly, will raise an error. Expects a real, finite-valued 1D or 2D
                ``numpy`` :obj:`np.ndarray`-typed array.

        Warnings:
            :attr:`audio_data` and :attr:`stft_data` are not automatically synchronized, meaning
            that if one of them is changed, those changes are not instantly reflected in the other.
            To propagate changes, either call :func:`stft` or :func:`istft`.


        Notes:
            * This attribute only returns values within the active region. For more information
                see :func:`set_active_region_to_default`. When setting this attribute, the active
                region are reset to default.

            * If :attr:`audio_data` is set with an improperly transposed array, it will
                automatically transpose it so that it is set the expected way. A warning will be
                displayed on the console.

        See Also:
            * :func:`load_audio_from_file` to load audio into :attr:`audio_data` after
                initialization.

            * :func:`load_audio_from_array` to safely load audio into :attr:`audio_data` after
                initialization.

            * :func:`set_active_region_to_default` for more information about the active region.

            * :attr:`signal_duration` and :attr:`signal_length` for length of audio data in seconds
                and samples, respectively.

            * :func:`stft` to calculate an STFT from this data,
                and :func:`istft` to calculate the inverse STFT and put it in :attr:`audio_data`.

            * :func:`plot_time_domain` to create a plot of audio data stored in this attribute.

            * :func:`peak_normalize` to apply gain such that to the absolute max value is exactly
                ``1.0``.

            * :func:`rms` to calculate the root-mean-square of :attr:`audio_data`

            * :func:`apply_gain` to apply a gain.

            * :func:`get_channel` to safely retrieve a single channel in :attr:`audio_data`.
        """
        if self._audio_data is None:
            return None

        start = 0
        end = self._audio_data.shape[constants.LEN_INDEX]

        if self._active_end is not None and self._active_end < end:
            end = self._active_end

        if self._active_start is not None and self._active_start > 0:
            start = self._active_start

        return self._audio_data[:, start:end]

    @audio_data.setter
    def audio_data(self, value):

        if value is None:
            self._audio_data = None
            return

        elif not isinstance(value, np.ndarray):
            raise AudioSignalException('Type of self.audio_data must be of type np.ndarray!')

        if not np.isfinite(value).all():
            raise AudioSignalException('Not all values of audio_data are finite!')

        if value.ndim > 1 and value.shape[constants.CHAN_INDEX] > value.shape[constants.LEN_INDEX]:
            value = value.T

        if value.ndim > 2:
            raise AudioSignalException('self.audio_data cannot have more than 2 dimensions!')

        if value.ndim < 2:
            value = np.expand_dims(value, axis=constants.CHAN_INDEX)

        self._audio_data = value

        self.set_active_region_to_default()

    @property
    def stft_data(self):
        """
        ``np.ndarray``
            Stored as a ``numpy`` :obj:`np.ndarray`, :attr:`stft_data` houses complex-valued data
            computed from a Short-time Fourier Transform (STFT) of audio data in the
            :class:`AudioSignal`. ``None`` by default, this :class:`AudioSignal` object can be
            initialized with STFT data upon initialization or it can be set at any time.

            The STFT data is stored with shape ``(n_frequency_bins, n_hops, n_channels)`` as
            a complex-valued ``numpy`` array.

        Raises:
            :class:`AudioSignalException`
                if set with an :obj:`np.ndarray` with one dimension or more than three dimensions.

        See Also:
            * :func:`stft` to calculate an STFT from :attr:`audio_data`, and :func:`istft` to
             calculate the inverse STFT from this attribute and put it in :attr:`audio_data`.

            * :func:`magnitude_spectrogram` to calculate and get the magnitude spectrogram from
             :attr:`stft_data`. :func:`power_spectrogram` to calculate and get the power
             spectrogram from :attr:`stft_data`.

            * :func:`get_stft_channel` to safely get a specific channel in :attr:`stft_data`.

        Notes:
            * :attr:`audio_data` and :attr:`stft_data` are not automatically synchronized, meaning
            that if one of them is changed, those changes are not instantly reflected in the other.
            To propagate changes, either call :func:`stft` or :func:`istft`.

            * :attr:`stft_data` will expand a two dimensional array so that it has the expected
            shape `(n_frequency_bins, n_hops, n_channels)`.
        """

        return self._stft_data

    @stft_data.setter
    def stft_data(self, value):
        if value is None:
            self._stft_data = None
            return

        elif not isinstance(value, np.ndarray):
            raise AudioSignalException('Type of self.stft_data must be of type np.ndarray!')

        if value.ndim == 1:
            raise AudioSignalException('Cannot support arrays with less than 2 dimensions!')

        if value.ndim == 2:
            value = np.expand_dims(value, axis=constants.STFT_CHAN_INDEX)

        if value.ndim > 3:
            raise AudioSignalException('Cannot support arrays with more than 3 dimensions!')

        if not np.iscomplexobj(value):
            warnings.warn('Initializing STFT with data that is non-complex. '
                          'This might lead to weird results!')

        self._stft_data = value

    @property
    def stft_params(self):
        """
        ``STFTParams``
            STFT parameters are kept in this property. STFT parameters are a ``namedtuple``
            called ``STFTParams`` with the following signature:

            .. code-block:: python

                STFTParams(
                    window_length=2048,
                    hop_length=512,
                    window_type='hann'
                )

            The defaults are 32ms windows, 8ms hop, and a hann window.

        """
        return self._stft_params

    @stft_params.setter
    def stft_params(self, value):
        if value and not isinstance(value, STFTParams):
            raise ValueError("stft_params must be of type STFTParams or None!")

        default_win_len = int(
            2 ** (np.ceil(np.log2(constants.DEFAULT_WIN_LEN_PARAM * self.sample_rate)))
        )
        default_hop_len = default_win_len // 4
        default_win_type = constants.WINDOW_DEFAULT

        default_stft_params = STFTParams(
            window_length=default_win_len,
            hop_length=default_hop_len,
            window_type=default_win_type
        )._asdict()

        value = value._asdict() if value else default_stft_params

        for key in default_stft_params:
            if value[key] is None:
                value[key] = default_stft_params[key]

        self._stft_params = STFTParams(**value)
        if self._stft_params.window_type == 'sqrt_hann':
            window_type = constants.WINDOW_HANN
        else:
            window_type = self._stft_params.window_type
        check_COLA(window_type, self._stft_params.window_length, self._stft_params.hop_length)

    @property
    def has_data(self):
        """
        ``bool``
            Returns ``False`` if :attr:`audio_data` and :attr:`stft_data` are empty. Else,
            returns ``True``.
        """
        has_audio_data = self.audio_data is not None and self.audio_data.size != 0
        has_stft_data = self.stft_data is not None and self.stft_data.size != 0
        return has_audio_data or has_stft_data

    @property
    def file_name(self):
        """
        ``str``
            The name of the file associated with this object. Includes extension, but not the full
            path.
        
        Notes:
            This will return ``None`` if this :class:`AudioSignal` object was not
            loaded from a file.
        
        See Also:
            :attr:`path_to_input_file` for the full path.
        """
        if self.path_to_input_file is not None:
            return os.path.basename(self.path_to_input_file)
        return None

    @property
    def sample_rate(self):
        """
        ``int``
            Sample rate associated with this object. If audio was read from a file, the sample
            rate will be set to the sample rate associated with the file. If this object was
            initialized from an array then the sample rate is set upon init. This property is
            read-only. To change the sample rate, use :func:`resample`.

        Notes:
            This property is read-only and cannot be set directly. To change

        See Also:
            * :func:`resample` to change the sample rate and resample data in :attr:`sample_rate`.

            * :func:`load_audio_from_array` to read audio from an array and set the sample rate.

            * :var:`nussl.constants.DEFAULT_SAMPLE_RATE` the default sample rate for *nussl*
                if not specified
        """
        return self._sample_rate

    @property
    def time_vector(self):
        """
        ``np.ndarray``
            A 1D :obj:`np.ndarray` with timestamps (in seconds) for each sample in
            :attr:`audio_data`.
        """
        if self.signal_duration is None:
            return None
        return np.linspace(0.0, self.signal_duration, num=self.signal_length)

    @property
    def freq_vector(self):
        """
        ``np.ndarray``
            A 1D numpy array with frequency values (in Hz) that correspond
            to each frequency bin (vertical axis) in :attr:`stft_data`. Assumes
            linearly spaced frequency bins.

        Raises:
            :class:`AudioSignalException`: If :attr:`stft_data` is ``None``. 
                Run :func:`stft` before accessing this.
        """
        if self.stft_data is None:
            raise AudioSignalException(
                'Cannot calculate freq_vector until self.stft() is run')
        return np.linspace(
            0.0, self.sample_rate // 2,
            num=self.stft_data.shape[constants.STFT_VERT_INDEX])

    @property
    def time_bins_vector(self):
        """
        ``np.ndarray``
            A 1D numpy array with time values (in seconds) that correspond
            to each time bin (horizontal/time axis) in :attr:`stft_data`.

        Raises:
            :class:`AudioSignalException`: If :attr:`stft_data` is ``None``. Run :func:`stft`
                before accessing this.
        """
        if self.stft_data is None:
            raise AudioSignalException(
                'Cannot calculate time_bins_vector until self.stft() is run')
        return np.linspace(0.0, self.signal_duration,
                           num=self.stft_data.shape[constants.STFT_LEN_INDEX])

    @property
    def stft_length(self):
        """
        ``int``
            The length of :attr:`stft_data` along the time axis. In units of hops.

        Raises:
            :class:`AudioSignalException`: If ``self.stft_dat``a is ``None``. Run :func:`stft`
                before accessing this.
        """
        if self.stft_data is None:
            raise AudioSignalException('Cannot calculate stft_length until self.stft() is run')
        return self.stft_data.shape[constants.STFT_LEN_INDEX]

    @property
    def active_region_is_default(self):
        """
        ``bool``
            ``True`` if active region is the full length of :attr:`audio_data`. ``False`` otherwise.

        See Also:

            * :func:`set_active_region` for a description of active regions in :class:`AudioSignal`

            * :func:`set_active_region_to_default`
        """
        return self._active_start == 0 and self._active_end == self._signal_length

    @property
    def _signal_length(self):
        """
        ``int``
            This is the length of the full signal, not just the active region.

        """
        if self._audio_data is None:
            return None
        return self._audio_data.shape[constants.LEN_INDEX]

    @property
    def power_spectrogram_data(self):
        """
        ``np.ndarray``
            Returns a real valued :obj:`np.ndarray` with power
            spectrogram data. The power spectrogram is defined as ``(STFT)^2``, where ``^2`` is
            element-wise squaring of entries of the STFT. Same shape as :attr:`stft_data`.
        
        Raises:
            :class:`AudioSignalException`: if :attr:`stft_data` is ``None``. Run :func:`stft`
                before accessing this.
            
        See Also:
            * :func:`stft` to calculate the STFT before accessing this attribute.
            * :attr:`stft_data` complex-valued Short-time Fourier Transform data.
            * :attr:`magnitude_spectrogram_data` to get magnitude spectrogram data.
            * :func:`get_power_spectrogram_channel` to get a specific channel
            
        """
        if self.stft_data is None:
            raise AudioSignalException('Cannot calculate power_spectrogram_data '
                                       'because self.stft_data is None')
        return np.abs(self.stft_data) ** 2

    @property
    def magnitude_spectrogram_data(self):
        """
        ``np.ndarray``
            Returns a real valued ``np.array`` with magnitude spectrogram data. The magnitude
            spectrogram is defined as ``abs(STFT)``, the element-wise absolute value of every item
            in the STFT. Same shape as :attr:`stft_data`.
        
        Raises:
            AudioSignalException: if :attr:`stft_data` is ``None``. Run :func:`stft` before
                accessing this.
            
        See Also:
            * :func:`stft` to calculate the STFT before accessing this attribute.
            * :attr:`stft_data` complex-valued Short-time Fourier Transform data.
            * :attr:`power_spectrogram_data`
            * :func:`get_magnitude_spectrogram_channel`
            
        """
        if self.stft_data is None:
            raise AudioSignalException('Cannot calculate magnitude_spectrogram_data '
                                       'because self.stft_data is None')
        return np.abs(self.stft_data)

    @property
    def log_magnitude_spectrogram_data(self):
        """
        (:obj:`np.ndarray`): Returns a real valued ``np.array`` with log magnitude spectrogram data.
        
        The log magnitude spectrogram is defined as 20 * log10(abs(stft)).
        Same shape as :attr:`stft_data`.
        
        Raises:
            AudioSignalException: if :attr:`stft_data` is ``None``. Run :func:`stft` before
                accessing this.
            
        See Also:
            * :func:`stft` to calculate the STFT before accessing this attribute.
            * :attr:`stft_data` complex-valued Short-time Fourier Transform data.
            * :attr:`power_spectrogram_data`
            * :func:`get_magnitude_spectrogram_channel`
            
        """
        if self.stft_data is None:
            raise AudioSignalException('Cannot calculate log_magnitude_spectrogram_data '
                                       'because self.stft_data is None')
        return 20 * np.log10(np.abs(self.stft_data) + 1e-8)
    
    @property
    def effects_chain(self):
        """
        (``list`` of ``nussl.core.FilterFunction``): Returns a copy of the AudioSignal's
        effect chain. Editing this property will not result in a change to the effects chain
        of the AudioSignal. 

        Please use the effects hooks (e.g. :func:`tremolo`, :func:`make_effect`) to make changes
        to the Audiosignal's effects chain.

        See Also:
            * :func:`apply_effects`
        """
        return self._effects_chain.copy()

    @property
    def effects_applied(self):
        """
        (``list`` of ``nussl.core.FilterFunction``): Returns a copy of the list of effects 
        applied to the AudioSignal. Editing this property will not result in a change to the 
        effects aplied to the AudioSignal. 

        Please use :func:`apply_effects` to apply effects to the AudioSignal.
         
        See Also:
            * :func:`apply_effects`
        """
        return self._effects_applied.copy()

    ##################################################
    #                     I/O
    ##################################################

    def load_audio_from_file(self, input_file_path, offset=0, duration=None, new_sample_rate=None):
        # type: (str, float, float, int) -> None
        """
        Loads an audio signal into memory from a file on disc. The audio is stored in
        :class:`AudioSignal` as a :obj:`np.ndarray` of `float` s. The sample rate is read from
        the file, and this :class:`AudioSignal` object's sample rate is set from it. If
        :param:`new_sample_rate` is not ``None`` nor the same as the sample rate of the file,
        the audio will be resampled to the sample rate provided in the :param:`new_sample_rate`
        parameter. After reading the audio data into memory, the active region is set to default.

        :param:`offset` and :param:`duration` allow the user to determine how much of the audio is
        read from the file. If those are non-default, then only the values provided will be stored
        in :attr:`audio_data` (unlike with the active region, which has the entire audio data stored
        in memory but only allows access to a subset of the audio).

        See Also:
            * :func:`load_audio_from_array` to read audio data from a :obj:`np.ndarray`.

        Args:
            input_file_path (str): Path to input file.
            offset (float,): The starting point of the section to be extracted (seconds).
                Defaults to 0 seconds (i.e., the very beginning of the file).
            duration (float): Length of signal to load in second.
                signal_length of 0 means read the whole file. Defaults to the full
                length of the signal.
            new_sample_rate (int): If this parameter is not ``None`` or the same sample rate as
                provided by the input file, then the audio data will be resampled to the new
                sample rate dictated by this parameter.

        """
        assert offset >= 0, 'Parameter `offset` must be >= 0!'
        if duration is not None:
            assert duration >= 0, 'Parameter `duration` must be >= 0!'

        try:
            # try reading headers with soundfile for speed
            audio_info = sf.info(input_file_path)
            file_length = audio_info.duration
        except:
            # if that doesn't work try audioread
            with audioread.audio_open(os.path.realpath(input_file_path)) as input_file:
                file_length = input_file.duration

        if offset > file_length:
            raise AudioSignalException('offset is longer than signal!')

        if duration is not None and offset + duration >= file_length:
            warnings.warn('offset + duration are longer than the signal.'
                          ' Reading until end of signal...',
                          UserWarning)

        audio_input, self._sample_rate = librosa.load(input_file_path,
                                                      sr=None,
                                                      offset=offset,
                                                      duration=duration,
                                                      mono=False)

        self.audio_data = audio_input
        self.original_signal_length = self.signal_length

        if new_sample_rate is not None and new_sample_rate != self._sample_rate:
            warnings.warn('Input sample rate is different than the sample rate'
                          ' read from the file! Resampling...',
                          UserWarning)
            self.resample(new_sample_rate)

        self.path_to_input_file = input_file_path
        self.set_active_region_to_default()

    def load_audio_from_array(self, signal, sample_rate=constants.DEFAULT_SAMPLE_RATE):
        """
        Loads an audio signal from a :obj:`np.ndarray`. :param:`sample_rate` is the sample
        of the signal.

        See Also:
            * :func:`load_audio_from_file` to read in an audio file from disc.

        Notes:
            Only accepts float arrays and int arrays of depth 16-bits.

        Parameters:
            signal (:obj:`np.ndarray`): Array containing the audio signal sampled at
                :param:`sample_rate`.
            sample_rate (int): The sample rate of signal.
                Default is :ref:`constants.DEFAULT_SAMPLE_RATE` (44.1kHz)

        """
        assert (type(signal) == np.ndarray)

        self.path_to_input_file = None

        # Change from fixed point to floating point
        if not np.issubdtype(signal.dtype, np.floating):
            signal = signal.astype('float') / (np.iinfo(np.dtype('int16')).max + 1.0)

        self.audio_data = signal
        self.original_signal_length = self.signal_length
        self._sample_rate = sample_rate if sample_rate is not None \
            else constants.DEFAULT_SAMPLE_RATE

        self.set_active_region_to_default()

    def write_audio_to_file(self, output_file_path, sample_rate=None):
        """
        Outputs the audio signal data in :attr:`audio_data` to a file at :param:`output_file_path`
        with sample rate of :param:`sample_rate`.

        Parameters:
            output_file_path (str): Filename where output file will be saved.
            sample_rate (int): The sample rate to write the file at. Default is
                :attr:`sample_rate`.
        """
        if self.audio_data is None:
            raise AudioSignalException("Cannot write audio file because there is no audio data.")

        if sample_rate is None:
            sample_rate = self.sample_rate

        audio_output = np.copy(self.audio_data)

        # TODO: better fix
        # convert to fixed point again
        if not np.issubdtype(audio_output.dtype, np.dtype(int).type):
            audio_output = np.multiply(
                audio_output,
                2 ** (constants.DEFAULT_BIT_DEPTH - 1)).astype('int16')
        wav.write(output_file_path, sample_rate, audio_output.T)

    ##################################################
    #                Active Region
    ##################################################

    def set_active_region(self, start, end):
        """
        Determines the bounds of what gets returned when you access :attr:`audio_data`.
        None of the data in :attr:`audio_data` is discarded when you set the active region, it
        merely becomes inaccessible until the active region is set back to default (i.e., the full
        length of the signal).

        This is useful for reusing a single :class:`AudioSignal` object to do multiple operations on
        only select parts of the audio data.

        Warnings:
            Many functions will raise exceptions while the active region is not default. Be aware
            that adding, subtracting, concatenating, truncating, and other utilities are not
            available when the active region is not default.

        See Also:
            * :func:`set_active_region_to_default`
            * :attr:`active_region_is_default`

        Examples:
            >>> import nussl
            >>> import numpy as np
            >>> n = nussl.constants.DEFAULT_SAMPLE_RATE  # 1 second of audio at 44.1kHz
            >>> np_sin = np.sin(np.linspace(0, 100 * 2 * np.pi, n))  # sine wave @ 100 Hz
            >>> sig = nussl.AudioSignal(audio_data_array=np_sin)
            >>> sig.signal_duration
            1.0
            >>> sig.set_active_region(0, n // 2)
            >>> sig.signal_duration
            0.5

        Args:
            start (int): Beginning of active region (in samples). Cannot be less than 0.
            end (int): End of active region (in samples). Cannot be larger than
                :attr:`signal_length`.

        """
        start, end = int(start), int(end)
        self._active_start = start if start >= 0 else 0
        self._active_end = end if end < self._signal_length else self._signal_length

    def set_active_region_to_default(self):
        """
        Resets the active region of this :class:`AudioSignal` object to its default value of the
        entire :attr:`audio_data` array.
        
        See Also:
            * :func:`set_active_region` for an explanation of active regions within the
            :class:`AudioSignal`.

        """
        self._active_start = 0
        self._active_end = self._signal_length

    ##################################################
    #               STFT Utilities
    ##################################################

    @staticmethod
    def get_window(window_type, window_length):
        """
        Wrapper around scipy.signal.get_window so one can also get the 
        popular sqrt-hann window.
        
        Args:
            window_type (str): Type of window to get (see constants.ALL_WINDOW).
            window_length (int): Length of the window
        
        Returns:
            np.ndarray: Window returned by scipy.signa.get_window
        """
        if window_type == constants.WINDOW_SQRT_HANN:
            window = np.sqrt(scipy.signal.get_window(
                'hann', window_length
            ))
        else:
            window = scipy.signal.get_window(
                window_type, window_length)

        return window

    def stft(self, window_length=None, hop_length=None, window_type=None, overwrite=True):
        """
        Computes the Short Time Fourier Transform (STFT) of :attr:`audio_data`.
        The results of the STFT calculation can be accessed from :attr:`stft_data`
        if :attr:`stft_data` is ``None`` prior to running this function or ``overwrite == True``

        Warning:
            If overwrite=True (default) this will overwrite any data in :attr:`stft_data`!

        Args:
            window_length (int): Amount of time (in samples) to do an FFT on
            hop_length (int): Amount of time (in samples) to skip ahead for the new FFT
            window_type (str): Type of scaling to apply to the window.
            overwrite (bool): Overwrite :attr:`stft_data` with current calculation

        Returns:
            (:obj:`np.ndarray`) Calculated, complex-valued STFT from :attr:`audio_data`, 3D numpy
            array with shape `(n_frequency_bins, n_hops, n_channels)`.

        """
        if self.audio_data is None or self.audio_data.size == 0:
            raise AudioSignalException(
                "No time domain signal (self.audio_data) to make STFT from!")

        window_length = (
            self.stft_params.window_length
            if window_length is None
            else int(window_length)
        )
        hop_length = (
            self.stft_params.hop_length
            if hop_length is None
            else int(hop_length)
        )
        window_type = (
            self.stft_params.window_type
            if window_type is None
            else window_type
        )

        stft_data = []

        window = self.get_window(window_type, window_length)

        for chan in self.get_channels():
            _, _, _stft = scipy.signal.stft(
                chan, fs=self.sample_rate, window=window,
                nperseg=window_length, noverlap=window_length - hop_length)
            stft_data.append(_stft)

        stft_data = np.array(stft_data).transpose((1, 2, 0))

        if overwrite:
            self.stft_data = stft_data

        return stft_data

    def istft(self, window_length=None, hop_length=None, window_type=None, overwrite=True,
              truncate_to_length=None):
        """ Computes and returns the inverse Short Time Fourier Transform (iSTFT).

        The results of the iSTFT calculation can be accessed from :attr:`audio_data`
        if :attr:`audio_data` is ``None`` prior to running this function or ``overwrite == True``

        Warning:
            If overwrite=True (default) this will overwrite any data in :attr:`audio_data`!

        Args:
            window_length (int): Amount of time (in samples) to do an FFT on
            hop_length (int): Amount of time (in samples) to skip ahead for the new FFT
            window_type (str): Type of scaling to apply to the window.
            overwrite (bool): Overwrite :attr:`stft_data` with current calculation
            truncate_to_length (int): truncate resultant signal to specified length. Default ``None``.

        Returns:
            (:obj:`np.ndarray`) Calculated, real-valued iSTFT from :attr:`stft_data`, 2D numpy array
            with shape `(n_channels, n_samples)`.

        """
        if self.stft_data is None or self.stft_data.size == 0:
            raise AudioSignalException('Cannot do inverse STFT without self.stft_data!')

        window_length = (
            self.stft_params.window_length
            if window_length is None
            else int(window_length)
        )
        hop_length = (
            self.stft_params.hop_length
            if hop_length is None
            else int(hop_length)
        )
        window_type = (
            self.stft_params.window_type
            if window_type is None
            else window_type
        )

        signals = []

        window = self.get_window(window_type, window_length)

        for stft in self.get_stft_channels():
            _, _signal = scipy.signal.istft(
                stft, fs=self.sample_rate, window=window,
                nperseg=window_length, noverlap=window_length - hop_length)

            signals.append(_signal)

        calculated_signal = np.array(signals)

        # Make sure it's shaped correctly
        calculated_signal = np.expand_dims(calculated_signal, -1) \
            if calculated_signal.ndim == 1 else calculated_signal

        # if truncate_to_length isn't provided
        if truncate_to_length is None:
            truncate_to_length = self.original_signal_length
            if self.signal_length is not None:
                truncate_to_length = self.signal_length

        if truncate_to_length is not None and truncate_to_length > 0:
            calculated_signal = calculated_signal[:, :truncate_to_length]

        if overwrite or self.audio_data is None:
            self.audio_data = calculated_signal

        return calculated_signal

    def apply_mask(self, mask, overwrite=False):
        """
        Applies the input mask to the time-frequency representation in this :class:`AudioSignal`
        object and returns a new :class:`AudioSignal` object with the mask applied. The mask
        is applied to the magnitude of audio signal. The phase of the original audio
        signal is then applied to construct the masked STFT.
        
        Args:
            mask (:obj:`MaskBase`-derived object): A ``MaskBase``-derived object 
                containing a mask.
            overwrite (bool): If ``True``, this will alter ``stft_data`` in self. 
                If ``False``, this function will create a new ``AudioSignal`` object 
                with the mask applied.

        Returns:
            A new :class:`AudioSignal`` object with the input mask applied to the STFT,
            iff ``overwrite`` is False.

        """
        if not isinstance(mask, masks.MaskBase):
            raise AudioSignalException(f'Expected MaskBase-derived object, given {type(mask)}')

        if self.stft_data is None:
            raise AudioSignalException('There is no STFT data to apply a mask to!')

        if mask.shape != self.stft_data.shape:
            if not mask.shape[:-1] == self.stft_data.shape[:-1]:
                raise AudioSignalException(
                    'Input mask and self.stft_data are not the same shape! mask:'
                    f' {mask.shape}, self.stft_data: {self.stft_data.shape}'
                )

        magnitude, phase = np.abs(self.stft_data), np.angle(self.stft_data)
        masked_abs = magnitude * mask.mask
        masked_stft = masked_abs * np.exp(1j * phase)

        if overwrite:
            self.stft_data = masked_stft
        else:
            return self.make_copy_with_stft_data(masked_stft, verbose=False)

    def ipd_ild_features(self, ch_one=0, ch_two=1):
        """
        Computes interphase difference (IPD) and interlevel difference (ILD) for a 
        stereo spectrogram. If more than two channels, this by default computes IPD/ILD
        between the first two channels. This can be specified by the arguments ch_one
        and ch_two. If only one channel, this raises an error.
        
        Args:
            ch_one (``int``): index of first channel to compute IPD/ILD.
            ch_two (``int``): index of second channel to compute IPD/ILD.

        Returns:
            ipd (``np.ndarray``): Interphase difference between selected channels
            ild (``np.ndarray``): Interlevel difference between selected channels

        """
        if self.stft_data is None:
            raise AudioSignalException("Cannot compute ipd/ild features without stft_data!")
        if self.is_mono:
            raise AudioSignalException("Cannot compute ipd/ild features on mono input!")

        stft_ch_one = self.get_stft_channel(ch_one)
        stft_ch_two = self.get_stft_channel(ch_two)

        ild = np.abs(stft_ch_one) / (np.abs(stft_ch_two) + 1e-4)
        ild = 20 * np.log10(ild + 1e-8)

        frequencies = self.freq_vector
        ipd = np.angle(stft_ch_two * np.conj(stft_ch_one))
        ipd /= (frequencies + 1.0)[:, None]
        ipd = ipd % np.pi

        return ipd, ild
    

    ##################################################
    #                  Utilities
    ##################################################

    def concat(self, other):
        """ Concatenate two :class:`AudioSignal` objects (by concatenating :attr:`audio_data`).

        Puts ``other.audio_data`` after :attr:`audio_data`.

        Raises:
            AudioSignalException: If ``self.sample_rate != other.sample_rate``,
                ``self.num_channels != other.num_channels``, or ``!self.active_region_is_default``
                is ``False``.

        Args:
            other (:class:`AudioSignal`): :class:`AudioSignal` to concatenate with the current one.
            
        """
        self._verify_audio(other)

        self.audio_data = np.concatenate((self.audio_data, other.audio_data),
                                         axis=constants.LEN_INDEX)

    def truncate_samples(self, n_samples):
        """ Truncates the signal leaving only the first ``n_samples`` samples.
        This can only be done if ``self.active_region_is_default`` is True. If
        ``n_samples > self.signal_length``, then `n_samples = self.signal_length` 
        (no truncation happens).

        Raises:
            AudioSignalException: If ``self.active_region_is_default`` is ``False``.

        Args:
            n_samples: (int) number of samples that will be left.

        """
        if not self.active_region_is_default:
            raise AudioSignalException('Cannot truncate while active region is not set as default!')

        n_samples = int(n_samples)
        if n_samples > self.signal_length:
            n_samples = self.signal_length

        self.audio_data = self.audio_data[:, 0: n_samples]

    def truncate_seconds(self, n_seconds):
        """ Truncates the signal leaving only the first n_seconds.
        This can only be done if self.active_region_is_default is True.

        Args:
            n_seconds: (float) number of seconds to truncate :attr:`audio_data`.

        """
        n_samples = int(n_seconds * self.sample_rate)
        self.truncate_samples(n_samples)

    def crop_signal(self, before, after):
        """
        Get rid of samples before and after the signal on all channels. Contracts the length
        of :attr:`audio_data` by before + after. Useful to get rid of zero padding after the fact.

        Args:
            before: (int) number of samples to remove at beginning of self.audio_data
            after: (int) number of samples to remove at end of self.audio_data

        """
        if not self.active_region_is_default:
            raise AudioSignalException('Cannot crop signal while active region '
                                       'is not set as default!')
        num_samples = self.signal_length
        self.audio_data = self.audio_data[:, before:num_samples - after]
        self.set_active_region_to_default()

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
            raise AudioSignalException('Cannot zero-pad while active region is not set as default!')

        self.audio_data = np.pad(self.audio_data, ((0, 0), (before, after)), 'constant')

    def add(self, other):
        """Adds two audio signal objects.

        This does element-wise addition on the :attr:`audio_data` array.

        Raises:
            AudioSignalException: If ``self.sample_rate != other.sample_rate``,
                ``self.num_channels != other.num_channels``, or
                ``self.active_region_is_default`` is ``False``.

        Parameters:
            other (:class:`AudioSignal`): Other :class:`AudioSignal` to add.

        Returns:
            (:class:`AudioSignal`): New :class:`AudioSignal` object with the sum of
            ``self`` and ``other``.
        """
        if isinstance(other, int):
            # this is so that sum(list of audio_signals) works.
            # when sum is called on a list it's evaluated as 0 + elem1 + elem2 + ...
            # so the 0 case needs to be taken care of (by doing nothing)
            return self

        self._verify_audio_arithmetic(other)

        new_signal = copy.deepcopy(self)
        new_signal.audio_data = self.audio_data + other.audio_data

        return new_signal

    def subtract(self, other):
        """Subtracts two audio signal objects.

        This does element-wise subtraction on the :attr:`audio_data` array.

        Raises:
            AudioSignalException: If ``self.sample_rate != other.sample_rate``,
                ``self.num_channels != other.num_channels``, or
                ``self.active_region_is_default`` is ``False``.

        Parameters:
            other (:class:`AudioSignal`): Other :class:`AudioSignal` to subtract.

        Returns:
            (:class:`AudioSignal`): New :class:`AudioSignal` object with the difference
            between ``self`` and ``other``.
        """
        self._verify_audio_arithmetic(other)

        other_copy = copy.deepcopy(other)
        other_copy *= -1
        return self.add(other_copy)

    def make_copy_with_audio_data(self, audio_data, verbose=True):
        """ Makes a copy of this :class:`AudioSignal` object with :attr:`audio_data` initialized to
        the input :param:`audio_data` numpy array. The :attr:`stft_data` of the new
        :class:`AudioSignal`object is ``None``.

        Args:
            audio_data (:obj:`np.ndarray`): Audio data to be put into the new :class:`AudioSignal`
                object.
            verbose (bool): If ``True`` prints warnings. If ``False``, outputs nothing.

        Returns:
            (:class:`AudioSignal`): A copy of this :class:`AudioSignal` object with :attr:`audio_data`
            initialized to the input :param:`audio_data` numpy array.

        """
        if verbose:
            if not self.active_region_is_default:
                warnings.warn('Making a copy when active region is not default.')

            if audio_data.shape != self.audio_data.shape:
                warnings.warn('Shape of new audio_data does not match current audio_data.')

        new_signal = copy.deepcopy(self)
        new_signal.audio_data = audio_data
        new_signal.stft_data = None
        return new_signal

    def make_copy_with_stft_data(self, stft_data, verbose=True):
        """ Makes a copy of this :class:`AudioSignal` object with :attr:`stft_data` initialized to
        the input :param:`stft_data` numpy array. The :attr:`audio_data` of the new
        :class:`AudioSignal` object is ``None``.

        Args:
            stft_data (:obj:`np.ndarray`): STFT data to be put into the new :class:`AudioSignal`
                object.

        Returns:
            (:class:`AudioSignal`): A copy of this :class:`AudioSignal` object with :attr:`stft_data`
            initialized to the input :param:`stft_data` numpy array.

        """
        if verbose:
            if not self.active_region_is_default:
                warnings.warn('Making a copy when active region is not default.')

            if stft_data.shape != self.stft_data.shape:
                warnings.warn('Shape of new stft_data does not match current stft_data.')

        new_signal = copy.deepcopy(self)
        new_signal.stft_data = stft_data
        new_signal.original_signal_length = self.original_signal_length
        new_signal.audio_data = None
        return new_signal

    def loudness(self, filter_class='K-weighting', block_size=0.400):
        """
        Uses pyloudnorm to calculate loudness.

        Implementation of ITU-R BS.1770-4.
        Allows control over gating block size and frequency weighting filters for 
        additional control.

        Measure the integrated gated loudness of a signal.
        
        Uses the weighting filters and block size defined by the meter
        the integrated loudness is measured based upon the gating algorithm
        defined in the ITU-R BS.1770-4 specification. 
        Supports up to 5 channels and follows the channel ordering: 
        [Left, Right, Center, Left surround, Right surround]

        Args:
            filter_class (str):
              Class of weighting filter used.
              - 'K-weighting' (default)
              - 'Fenton/Lee 1'
              - 'Fenton/Lee 2'
              - 'Dash et al.'
            block_size (float):
              Gating block size in seconds. Defaults to 0.400.

        Returns:
            float: LUFS, Integrated gated loudness of the input 
              measured in dB LUFS.
        """

        # create BS.1770 meter
        meter = pyloudnorm.Meter(
            self.sample_rate, filter_class=filter_class, block_size=block_size)
        # measure loudness
        loudness = meter.integrated_loudness(self.audio_data.T) 
        return loudness

    def rms(self, win_len=None, hop_len=None):
        """ Calculates the root-mean-square of :attr:`audio_data`.
        
        Returns:
            (float): Root-mean-square of :attr:`audio_data`.

        """
        if win_len is not None:
            hop_len = win_len // 2 if hop_len is None else hop_len
            rms_func = lambda arr: librosa.feature.rms(arr, frame_length=win_len,
                                                       hop_length=hop_len)[0, :]
        else:
            rms_func = lambda arr: np.sqrt(np.mean(np.square(arr)))

        result = []
        for ch in self.get_channels():
            result.append(rms_func(ch))

        return np.squeeze(result)

    def peak_normalize(self):
        """
        Peak normalizes the audio signal.
        """
        self.apply_gain(1 / np.abs(self.audio_data).max())

    def apply_gain(self, value):
        """
        Apply a gain to :attr:`audio_data`

        Args:
            value (float): amount to multiply self.audio_data by

        Returns:
            (:class:`AudioSignal`): This :class:`AudioSignal` object with the gain applied.

        """
        if not isinstance(value, numbers.Real):
            raise AudioSignalException('Can only multiply/divide by a scalar!')

        self.audio_data = self.audio_data * value
        return self

    def resample(self, new_sample_rate, **kwargs):
        """
        Resample the data in :attr:`audio_data` to the new sample rate provided by
        :param:`new_sample_rate`. If the :param:`new_sample_rate` is the same as :attr:`sample_rate`
        then nothing happens.

        Args:
            new_sample_rate (int): The new sample rate of :attr:`audio_data`.
            kwargs: Keyword arguments to librosa.resample.

        """

        if new_sample_rate == self.sample_rate:
            warnings.warn('Cannot resample to the same sample rate.')
            return

        resampled_signal = []

        for channel in self.get_channels():
            resampled_channel = librosa.resample(
                channel, self.sample_rate, new_sample_rate, **kwargs)
            resampled_signal.append(resampled_channel)

        self.audio_data = np.array(resampled_signal)
        self.original_signal_length = self.signal_length
        self._sample_rate = new_sample_rate

    ##################################################
    #              Channel Utilities
    ##################################################

    def _verify_get_channel(self, n):
        if n >= self.num_channels:
            raise AudioSignalException(
                f'Cannot get channel {n} when this object only has {self.num_channels}'
                ' channels! (0-based)'
            )

        if n < 0:
            raise AudioSignalException(
                f'Cannot get channel {n}. This will cause unexpected results.'
            )

    def get_channel(self, n):
        """Gets audio data of n-th channel from :attr:`audio_data` as a 1D :obj:`np.ndarray`
        of shape ``(n_samples,)``.

        Parameters:
            n (int): index of channel to get. **0-based**

        See Also:
            * :func:`get_channels`: Generator for looping through channels of :attr:`audio_data`.
            * :func:`get_stft_channel`: Gets stft data from a specific channel.
            * :func:`get_stft_channels`: Generator for looping through channels from
            :attr:`stft_data`.

        Raises:
            :class:`AudioSignalException`: If not ``0 <= n < self.num_channels``.
            
        Returns:
            (:obj:`np.array`): The audio data in the n-th channel of the signal, 1D

        """
        self._verify_get_channel(n)

        return np.asfortranarray(utils._get_axis(self.audio_data, constants.CHAN_INDEX, n))

    def get_channels(self):
        """Generator that will loop through channels of :attr:`audio_data`.

        See Also:
            * :func:`get_channel`: Gets audio data from a specific channel.
            * :func:`get_stft_channel`: Gets stft data from a specific channel.
            * :func:`get_stft_channels`: Generator to loop through channels of :attr:`stft_data`.

        Yields:
            (:obj:`np.array`): The audio data in the next channel of this signal as a
            1D ``np.ndarray``.

        """
        for i in range(self.num_channels):
            yield self.get_channel(i)

    def get_stft_channel(self, n):
        """Returns STFT data of n-th channel from :attr:`stft_data` as a 2D ``np.ndarray``.

        Args:
            n: (int) index of stft channel to get. **0-based**

        See Also:
            * :func:`get_stft_channels`: Generator to loop through channels from :attr:`stft_data`.
            * :func:`get_channel`: Gets audio data from a specific channel.
            * :func:`get_channels`: Generator to loop through channels of :attr:`audio_data`.

        Raises:
            :class:`AudioSignalException`: If not ``0 <= n < self.num_channels``.

        Returns:
            (:obj:`np.array`): the STFT data in the n-th channel of the signal, 2D

        """
        if self.stft_data is None:
            raise AudioSignalException('Cannot get STFT data before STFT is calculated!')

        self._verify_get_channel(n)

        return utils._get_axis(self.stft_data, constants.STFT_CHAN_INDEX, n)

    def get_stft_channels(self):
        """Generator that will loop through channels of :attr:`stft_data`.

        See Also:
            * :func:`get_stft_channel`: Gets stft data from a specific channel.
            * :func:`get_channel`: Gets audio data from a specific channel.
            * :func:`get_channels`: Generator to loop through channels of :attr:`audio_data`.

        Yields:
            (:obj:`np.array`): The STFT data in the next channel of this signal as a
            2D ``np.ndarray``.

        """
        for i in range(self.num_channels):
            yield self.get_stft_channel(i)

    def make_audio_signal_from_channel(self, n):
        """
        Makes a new :class:`AudioSignal` object from with data from channel ``n``.
        
        Args:
            n (int): index of channel to make a new signal from. **0-based**

        Returns:
            (:class:`AudioSignal`) new :class:`AudioSignal` object with only data from
            channel ``n``.

        """
        new_signal = copy.copy(self)
        new_signal.audio_data = self.get_channel(n)
        return new_signal

    def get_power_spectrogram_channel(self, n):
        """ Returns the n-th channel from ``self.power_spectrogram_data``.

        Raises:
            Exception: If not ``0 <= n < self.num_channels``.

        Args:
            n: (int) index of power spectrogram channel to get **0-based**

        Returns:
            (:obj:`np.array`): the power spectrogram data in the n-th channel of the signal, 1D
        """
        self._verify_get_channel(n)

        # np.array helps with duck typing
        return utils._get_axis(np.array(self.power_spectrogram_data),
                               constants.STFT_CHAN_INDEX, n)

    def get_magnitude_spectrogram_channel(self, n):
        """ Returns the n-th channel from ``self.magnitude_spectrogram_data``.

        Raises:
           Exception: If not ``0 <= n < self.num_channels``.

        Args:
            n: (int) index of magnitude spectrogram channel to get **0-based**

        Returns:
            (:obj:`np.array`): the magnitude spectrogram data in the n-th channel of the signal, 1D
        """
        self._verify_get_channel(n)

        # np.array helps with duck typing
        return utils._get_axis(np.array(self.magnitude_spectrogram_data),
                               constants.STFT_CHAN_INDEX, n)

    def to_mono(self, overwrite=True, keep_dims=False):
        """ Converts :attr:`audio_data` to mono by averaging every sample.

        Args:
            overwrite (bool): If ``True`` this function will overwrite :attr:`audio_data`.
            keep_dims (bool): If ``False`` this function will return a 1D array,
                else will return array with shape `(1, n_samples)`.

        Warning:
            If ``overwrite=True`` (default) this will overwrite any data in :attr:`audio_data`!

        Returns:
            (:obj:`AudioSignal`): Mono-ed version of AudioSignal, either in place or not.

        """
        mono = np.mean(self.audio_data, axis=constants.CHAN_INDEX, keepdims=keep_dims)

        if overwrite:
            self.audio_data = mono
            return self
        else:
            mono_signal = self.make_copy_with_audio_data(mono)
            return mono_signal

    ##################################################
    #                 Utility hooks                  #
    ##################################################

    def play(self):
        """
        Plays this audio signal, using `nussl.play_utils.play`.

        Plays an audio signal if ffplay from the ffmpeg suite of tools is installed.
        Otherwise, will fail. The audio signal is written to a temporary file
        and then played with ffplay.
        """
        # lazy load
        from . import play_utils
        play_utils.play(self)

    def embed_audio(self, ext='.mp3', display=True):
        """
        Embeds the audio signal into a notebook, using `nussl.play_utils.embed_audio`.
        
        Write a numpy array to a temporary mp3 file using ffmpy, then embeds the mp3 
        into the notebook.

        Args:
            ext (str): What extension to use when embedding. '.mp3' is more lightweight 
            leading to smaller notebook sizes.
            display (bool): Whether or not to display the object immediately, or to return
            the html object for display later by the end user. Defaults to True.

        Example:
            >>> import nussl
            >>> audio_file = nussl.efz_utils.download_audio_file('schoolboy_fascination_excerpt.wav')
            >>> audio_signal = nussl.AudioSignal(audio_file)
            >>> audio_signal.embed_audio()

        This will show a little audio player where you can play the audio inline in 
        the notebook.        
        """
        # lazy load
        from . import play_utils
        return play_utils.embed_audio(self, ext=ext, display=display)

    ##################################
    #          Effect Hooks          #
    ##################################

    def apply_effects(self, reset=True, overwrite=False, user_order=True):
        """
        This method applies a prespecified set of audio effects (e.g., chorus, filtering,
        reverb, etc...) to this audio signal. Before any effect can be applied, the effects
        are first added to the "effects chain", which refers to a queue of effects that will be
        all applied to an AudioSignal object when this function is called. Effects are added
        to the effects chain through AudioSignal effect hooks which are `AudioSignal` methods for
        setting up an effect with the desired parameters. If the effect chain is empty this method
        does nothing. By default, when this method is called the effects chain empty. See the
        documentation below for a list of supported effects and their respective details.

        Notes:
            The effects will be added in the order that they are added to the effects chain, unless
            `user_order=False`, in case the order is not guaranteed to be preserved. Setting
            `user_order=False` will apply all SoX effects first, then FFMpeg effects, which can
            sped up processing time by ~30% in our experiments.

        Args:
            reset (bool): If True, clears out all effects in effect chains following applying the
                effects. Default=True
            overwrite (bool): If True, overwrites existing audio_data in AudioSignal. Default=False
                Also clears out `stft_data`.
            user_order (bool): If True, applied effects in the user provided order. If False,
                applies all SoX effects before all FFmpeg effects, which can be faster.
        Returns:
            self or new_signal (AudioSignal): If overwrite=True, then returns initially AudioSignal
            with edited audio_data. Otherwise, returns a new AudioSignal new_signal.

        Example:
            Here are some examples of demonstrating to apply effects to your audio signal. Let's
            start with an obvious effect, such as time stretching. We can add
            this effect to the effects chain by using the built-in effects hook, `time_stretch()`:

            >>> signal.signal_duration
            10.0
            >>> signal.time_stretch(0.5)
            >>> signal.signal_duration
            10.0

            You can find this effect in the AudioSignal's effects chain. 

            >>> effect = signal.effects_chain[0]
            >>> str(effect)
            "time_stretch (params: {factor=0.5})"

            However, the signal's duration hasn't changed! You will need to call `apply_effects()`
            to apply the changes in the signal's effects chains. Applied effects can be found in 
            `effects_applied`.

            >>> new_signal = signal.apply_effects()
            >>> new_signal.signal_duration
            5.0
            >>> str(new_signal.effects_applied[0])
            "time_stretch (params: {factor=0.5})"

            >>> # This doesn't change the original signal
            >>> signal.signal_duration
            10.0

            You can iterate through effects_chain to use the properties of FilterFunction
            objects as arguments to `make_effect`:

            >>> for effect in signal1.effects_applied:
            >>>     filter_ = effect.filter
            >>>     params = effect.params
            >>>     signal2.make_effect(filter_, **params)

            Using `apply_effects()` will clear out the current effects chain. This behavior can be
            avoided by setting `reset` to False. 

            >>> another_signal = signal.apply_effects()
            >>> another_signal.signal_duration
            10.0

            To clear out the current effects chain without applying effect, use
            `reset_effects_chain()`. It will not revert effects already applied (i.e., your audio
            will still have the effects you applied).

            If `apply_effects()` is called with empty effects chain, then it returns itself.

            >>> another_signal == signal
            True

            You can also chain effects together. This will add a tremolo effect followed by a high
            pass filter effect to the AudioSignal's effects chain (Note: order matters!):

            >>> audio_signal.tremolo(5, .6).high_pass(12000)

            Using overwrite here, we change the audio data of the variable `audio_signal`, rather
            than create a new signal:

            >>> audio_signal.apply_effects(overwrite=True)
            >>> audio_signal.effects_applied
            ["tremolo", "highpass"]

            If `user_order` is false, FFmpeg effects will be applied AFTER SoX effects, regardless
            of the order the hooks are applied. The effects `time_stretch` and `pitch_shift` are SoX
            effects. All others are FFmpeg effects. This may be done for speed, as applying all
            FFmpeg effects without interuption will be faster than being interrupted with a SoX
            effect.

            For example, the two statements will result in the same effected signal:

            >>> signal_1 = audio_signal.pitch_shift(4).tremolo(5, .6).apply_effects(user_order=False)
            >>> signal_2 = audio_signal.tremolo(5, .6).pitch_shift(4).apply_effects(user_order=False)
            >>> signal_1.effects_applied == signal_2.effects_applied
            True

            Refer to the specific documentation for each effect to determine whether it is a SoX
            effect or an FFmpeg effect.

        See Also:
            * :func:`reset_effects_chain`: Empties the effects chain without applying any effects.
            * :func:`time_stretch`: Changes the length without effecting the pitch.
            * :func:`pitch_shift`: Changes the pitch without effecting the length of the signal.
            * :func:`low_pass`: Applies a low pass filter to the signal.
            * :func:`high_pass`: Applies a high pass filter to the signal.
            * :func:`tremelo`: Applies a tremolo (volume wobbling) effect to the signal.
            * :func:`vibrato`: Applies a vibrato (pitch wobbling) effect to the signal.
            * :func:`chorus`: Applies a chorus effect to the signal.
            * :func:`phaser`: Applies a phaser effect to the signal.
            * :func:`flanger`: Applies a flanger effect to the signal.
            * :func:`emphasis`: Boosts certain frequency ranges of the signal.
            * :func:`compressor`: Compresses the dynamic range of the signal.
            * :func:`equalizer`: Applies an equalizer to the signal.
            * :func:`make_effect`: Syntactic sugar for adding an effect to the chain by name.
        """
        if user_order:
            new_signal = self._apply_user_ordered_effects()
        else:
            new_signal = self._apply_sox_ffmpeg_ordered_effects()
        new_signal.reset_effects_chain()
        if reset:
            self.reset_effects_chain()
        if overwrite:
            self.audio_data = new_signal.audio_data
            self._effects_applied += new_signal.effects_applied
            self.stft_data = None
            return self

        return new_signal

    def _apply_user_ordered_effects(self):
        new_signal = self
        i = j = 0

        while i < len(self._effects_chain):
            j += 1

            if j == len(self._effects_chain) or \
                type(self._effects_chain[i]) != type(self._effects_chain[j]):  # new fx type

                next_chain = self._effects_chain[i:j]
                if isinstance(next_chain[0], effects.SoXFilter):
                    new_signal = effects.apply_effects_sox(new_signal, next_chain)
                elif isinstance(next_chain[0], effects.FFmpegFilter):
                    new_signal = effects.apply_effects_ffmpeg(new_signal, next_chain)

                i = j

        return new_signal

    def _apply_sox_ffmpeg_ordered_effects(self):
        new_signal = self
        sox_effects_chain = []
        ffmpeg_effects_chain = []
        for f in self._effects_chain:
            if isinstance(f, effects.FFmpegFilter):
                ffmpeg_effects_chain.append(f)
            elif isinstance(f, effects.SoXFilter):
                sox_effects_chain.append(f)

        if sox_effects_chain:
            new_signal = effects.apply_effects_sox(new_signal, sox_effects_chain)
        if ffmpeg_effects_chain:
            new_signal = effects.apply_effects_ffmpeg(new_signal, ffmpeg_effects_chain)

        return new_signal

    def make_effect(self, effect, **kwargs):
        """
        Syntactic sugar for adding an arbitrary effect hook to the effects chain by name.

        Example:
            >>> signal.time_stretch(1.5)

            Is the same as
            >>> signal.make_effect("time_stretch", factor=1.5)

            The attributes of a FilterFunction in the lists effects_applied or effects_chain may 
            used with `make_effect`. 

            >>> for effect in signal1.effects_applied:
            >>>     filter_ = effect.filter
            >>>     params = effect.params
            >>>     signal2.make_effect(filter_, **params)

        Notes:
            This effect won't be applied until you call `apply_effect()`!

        Args:
            effect (str): Function name of desired effect hook of the AudioSignal
            **kwargs: Additional parameters for given effect. 
        Return:
            self: Initial AudioSignal with updated effect chains

        See Also:
            * :func:`apply_effects`: Applies effects once they are in the effects chain.
            * :func:`reset_effects_chain`: Empties the effects chain without applying any effects.
        """
        try:
            effect_hook = getattr(self, effect, None)
            effect_hook(**kwargs)
        except Exception as e:
            raise AudioSignalException(f"Error calling {effect} with parameters {kwargs}: `{e}`")
        
        return self

    def reset_effects_chain(self):
        """
        Clears effects chain of AudioSignal.

        This will not revert effects that have already been applied to the audio!

        See Also:
            * :func:`apply_effects`: Applies effects once they are in the effects chain.
            * :func:`make_effect`: Syntactic sugar for adding an effect to the chain by name.
        """
        self._effects_chain = []
        return self

    def time_stretch(self, factor, **kwargs):
        """
        Adds a time stretch filter to the AudioSignal's effects chain.
        A factor greater than one will shorten the signal, a factor less then one
        will lengthen the signal, and a factor of 1 will not change the signal.

        This is a SoX effect. Please see 
        https://pysox.readthedocs.io/en/latest/_modules/sox/transform.html#Transformer.tempo
        for details. 

        Notes:
            This effect won't be applied until you call `apply_effect()`!

        Args: 
            factor (float): Scaling factor for tempo change. Must be positive.
            kwargs: Arugments passed to `sox.transform.tempo`
        Returns:
            self: Initial AudioSignal with updated effect chains

        See Also:
            * :func:`apply_effects`: Applies effects once they are in the effects chain.
            * :func:`make_effect`: Syntactic sugar for adding an effect to the chain by name.
            * :func:`reset_effects_chain`: Empties the effects chain without applying any effects.
        """
        self._effects_chain.append(effects.time_stretch(factor, **kwargs))
        return self
    
    def pitch_shift(self, n_semitones, **kwargs):
        """
        Add pitch shift effect to AudioSignal's effect chain. 
        A positive shift will change the pitch of the signal by `n_semitones`
        semitones. If positive, pitch will get higher, if negative pitch will
        get lower.

        This is a SoX effect. Please see:
        https://pysox.readthedocs.io/en/latest/_modules/sox/transform.html#Transformer.pitch
        For details.

        Notes:
            This effect won't be applied until you call `apply_effect()`!

        Args: 
            n_semitones (float): The number of semitones to shift the audio.
                Positive values increases the frequency of the signal
            kwargs: Arugments passed to `sox.transform.pitch`
        Returns:
            self: Initial AudioSignal with updated effect chains

        See Also:
            * :func:`apply_effects`: Applies effects once they are in the effects chain.
            * :func:`make_effect`: Syntactic sugar for adding an effect to the chain by name.
            * :func:`reset_effects_chain`: Empties the effects chain without applying any effects.
        """
        self._effects_chain.append(effects.pitch_shift(n_semitones, **kwargs))
        return self
    
    def low_pass(self, freq, poles=2, width_type="h", width=0.707, **kwargs):
        """
        Add low pass effect to AudioSignal's effect chain

        This is a FFmpeg effect. Please see:
        https://ffmpeg.org/ffmpeg-all.html#lowpass
        for details.

        Notes:
            This effect won't be applied until you call `apply_effect()`!
        
        Args: 
            freq (float): Threshold for low pass. Should be positive
            poles (int): Number of poles. should be either 1 or 2
            width_type (str): Unit of width for filter. Must be either:
                'h': Hz
                'q': Q-factor
                'o': octave
                's': slope
                'k': kHz
            width (float): Band width in width_type units
            kwargs: Arguments passed to `ffmpeg.filter`
        Returns:
            self: Initial AudioSignal with updated effect chains

        See Also:
            * :func:`apply_effects`: Applies effects once they are in the effects chain.
            * :func:`make_effect`: Syntactic sugar for adding an effect to the chain by name.
            * :func:`reset_effects_chain`: Empties the effects chain without applying any effects.
        """
        self._effects_chain.append(effects.low_pass(freq, poles=poles,
                                                    width_type=width_type,
                                                    width=width, **kwargs))
        return self
    
    def high_pass(self, freq, poles=2, width_type="h", width=0.707, **kwargs):
        """
        Add high pass effect to AudioSignal's effect chain

        This is a FFmpeg effect. Please see:
        https://ffmpeg.org/ffmpeg-all.html#highpass
        for details.

        Notes:
            This effect won't be applied until you call `apply_effect()`!
        
        Args: 
            freq (float): Threshold for high pass. Should be positive scalar
            poles (int): Number of poles. should be either 1 or 2
            width_type (str): Unit of width for filter. Must be either:
                'h': Hz
                'q': Q-factor
                'o': octave
                's': slope
                'k': kHz
            width (float): Band width in width_type units
            kwargs: Arguments passed to `ffmpeg.filter`
        Returns:
            self: Initial AudioSignal with updated effect chains

        See Also:
            * :func:`apply_effects`: Applies effects once they are in the effects chain.
            * :func:`make_effect`: Syntactic sugar for adding an effect to the chain by name.
            * :func:`reset_effects_chain`: Empties the effects chain without applying any effects.
        """
        self._effects_chain.append(effects.high_pass(freq, poles=poles,
                                                     width_type=width_type,
                                                     width=width, **kwargs))
        return self

    def tremolo(self, mod_freq, mod_depth, **kwargs):
        """
        Add tremolo effect to AudioSignal's effect chain
        
        This is a FFmpeg effect. Please see
        https://ffmpeg.org/ffmpeg-all.html#tremolo
        for details.

        Notes:
            This effect won't be applied until you call `apply_effect()`!
        
        Args: 
            mod_freq (float): Modulation frequency. Must be between .1 and 20000.
            mod_depth (float): Modulation depth. Must be between 0 and 1.
            kwargs: Arguments passed to `ffmpeg.filter`
        Returns:
            self: Initial AudioSignal with updated effect chains

        See Also:
            * :func:`apply_effects`: Applies effects once they are in the effects chain.
            * :func:`make_effect`: Syntactic sugar for adding an effect to the chain by name.
            * :func:`reset_effects_chain`: Empties the effects chain without applying any effects.
        """
        self._effects_chain.append(effects.tremolo(mod_freq, mod_depth, **kwargs))
        return self
    
    def vibrato(self, mod_freq, mod_depth, **kwargs):
        """
        Add vibrato effect to AudioSignal's effect chain.

        This is a FFmpeg effect. Please see
        https://ffmpeg.org/ffmpeg-all.html#vibrato
        for details.

        Notes:
            This effect won't be applied until you call `apply_effect()`!
        
        Args: 
            mod_freq (float): Modulation frequency. Must be between .1 and 20000.
            mod_depth (float): Modulation depth. Must be between 0 and 1.
            kwargs: Arguments passed to `ffmpeg.filter`
        Returns:
            self: Initial AudioSignal with updated effect chains

        See Also:
            * :func:`apply_effects`: Applies effects once they are in the effects chain.
            * :func:`make_effect`: Syntactic sugar for adding an effect to the chain by name.
            * :func:`reset_effects_chain`: Empties the effects chain without applying any effects.
        """
        self._effects_chain.append(effects.vibrato(mod_freq, mod_depth, **kwargs))
        return self
    
    def chorus(self, delays, decays, speeds,
               depths, in_gain=0.4, out_gain=0.4, **kwargs):
        """
        Add chorus effect to AudioSignal's effect chain.

        This is a FFmpeg effect. Please see
        https://ffmpeg.org/ffmpeg-all.html#chorus
        for details.

        Notes:
            This effect won't be applied until you call `apply_effect()`!
        
        Args:
            delays (list of float): delays in ms. Typical Delay is 40ms-6ms
            decays (list of float): decays. Must be between 0 and 1
            speeds (list of float): speeds. Must be between 0 and 1
            depths (list of float): depths. Must be between 0 and 1
            in_gain (float): Proportion of input gain. Must be between 0 and 1
            out_gain (float): Proportion of output gain. Must be between 0 and 1
            kwargs: Arguments passed to `ffmpeg.filter`
        Returns:
            self: Initial AudioSignal with updated effect chains

        See Also:
            * :func:`apply_effects`: Applies effects once they are in the effects chain.
            * :func:`make_effect`: Syntactic sugar for adding an effect to the chain by name.
            * :func:`reset_effects_chain`: Empties the effects chain without applying any effects.
        """
        self._effects_chain.append(effects.chorus(delays, decays,
                                                  speeds, depths,
                                                  in_gain=in_gain,
                                                  out_gain=out_gain,
                                                  **kwargs))
        return self
        
    def phaser(self, in_gain=0.4, out_gain=0.74, delay=3, decay=0.4,
               speed=0.5, type_="triangular", **kwargs):
        """
        Add phaser effect to AudioSignal's effect chain

        This is a FFmpeg effect. Please see
        https://ffmpeg.org/ffmpeg-all.html#aphaser
        for details. 

        Notes:
            This effect won't be applied until you call `apply_effect()`!

        Args:
            in_gain (float): Proportion of input gain. Must be between 0 and 1
            out_gain (float): Proportion of output gain. Must be between 0 and 1.
            delay (float): Delay of chorus filter in ms. (Time between original signal and delayed)
            decay (float): Decay of copied signal. Must be between 0 and 1.
            speed (float): Modulation speed of the delayed filter. 
            type_ (str): modulation type. Either Triangular or Sinusoidal
                "triangular" or "t" for Triangular
                "sinusoidal" of "s" for sinusoidal
            kwargs: Arguments passed to `ffmpeg.filter`
        Returns:
            self: Initial AudioSignal with updated effect chains

        See Also:
            * :func:`apply_effects`: Applies effects once they are in the effects chain.
            * :func:`make_effect`: Syntactic sugar for adding an effect to the chain by name.
            * :func:`reset_effects_chain`: Empties the effects chain without applying any effects.
        """
        fx = effects.phaser(in_gain=in_gain, out_gain=out_gain, delay=delay,
                            decay=decay, speed=speed, type_=type_, **kwargs)
        self._effects_chain.append(fx)
        return self
    
    def flanger(self, delay=0, depth=2, regen=0, width=71, speed=0.5,
                phase=25, shape="sinusoidal", interp="linear", **kwargs):
        """
        Add flanger effect to AudioSignal's effect chain
        This is a FFmpeg effect. Please see
        https://ffmpeg.org/ffmpeg-all.html#flanger
        for details. 

        Notes:
            This effect won't be applied until you call `apply_effect()`!

        Args:
            delay (float): Base delay in ms between original signal and copy.
                Must be between 0 and 30.
            depth (float): Sweep delay in ms. Must be between 0 and 10.
            regen (float): Percentage regeneration, or delayed signal feedback.
                Must be between -95 and 95.
            width (float): Percentage of delayed signal. Must be between 0 and 100.
            speed (float): Sweeps per second. Must be in .1 to 10
            shape (str): Swept wave shape, Must be "triangular" or "sinusoidal".
            phase (float): swept wave percentage-shift for multi channel. Must be between 0 and 100.
            interp (str): Delay Line interpolation. Must be "linear" or "quadratic".
            kwargs: Arguments passed to `ffmpeg.filter`
        Returns:
            self: Initial AudioSignal with updated effect chains

        See Also:
            * :func:`apply_effects`: Applies effects once they are in the effects chain.
            * :func:`make_effect`: Syntactic sugar for adding an effect to the chain by name.
            * :func:`reset_effects_chain`: Empties the effects chain without applying any effects.
        """
        fx = effects.flanger(delay=delay, depth=depth, regen=regen, width=width,
                             speed=speed, phase=phase, shape=shape, interp=interp,
                             **kwargs)
        self._effects_chain.append(fx)
        return self
    
    def emphasis(self, level_in, level_out, type_="col", mode='production', **kwargs):
        """
        Add emphasis effect to AudioSignal's effect chain. An emphasis filter boosts 
        frequency ranges the most susceptible to noise in a medium. When restoring
        sounds from such a medium, a de-emphasis filter is used to de-boost boosted 
        frequencies. 

        This is a FFmpeg effect. Please see
        https://ffmpeg.org/ffmpeg-all.html#aemphasis
        for details.

        Notes:
            This effect won't be applied until you call `apply_effect()`!

        Args:
            level_in (float): Input gain
            level_out (float): Output gain
            type_ (str): physical medium type to convert/deconvert from.
                Must be one of the following: 
                - "col": Columbia 
                - "emi": EMI
                - "bsi": BSI (78RPM)
                - "riaa": RIAA
                - "cd": CD (Compact Disk)
                - "50fm": 50µs FM
                - "75fm": 75µs FM 
                - "50kf": 50µs FM-KF 
                - "75kf": 75µs FM-KF 
            mode (str): Filter mode. Must be one of the following:
                - "reproduction": Apply de-emphasis filter
                - "production": Apply emphasis filter
            kwargs: Arguments passed to `ffmpeg.filter`
        Returns:
            self: Initial AudioSignal with updated effect chains

        See Also:
            * :func:`apply_effects`: Applies effects once they are in the effects chain.
            * :func:`make_effect`: Syntactic sugar for adding an effect to the chain by name.
            * :func:`reset_effects_chain`: Empties the effects chain without applying any effects.
        """
        self._effects_chain.append(effects.emphasis(level_in, level_out,
                                                    type_=type_, mode=mode,
                                                    **kwargs))
        return self
    
    def compressor(self, level_in, mode="downward", reduction_ratio=2,
                   attack=20, release=250, makeup=1, knee=2.8284, link="average",
                   detection="rms", mix=1, threshold=0.125, **kwargs):
        """
        Add compressor effect to AudioSignal's effect chain
    
        This is a FFmpeg effect. Please see
        https://ffmpeg.org/ffmpeg-all.html#acompressor
        for details.

        Notes:
            This effect won't be applied until you call `apply_effect()`!

        Args:
            level_in (float): Input Gain
            mode (str): Mode of compressor operation. Can either be "upward" or "downward". 
            threshold (float): Volume threshold. If a signal's volume is above the threshold,
                gain reduction would apply.
            reduction_ratio (float): Ratio in which the signal is reduced.
            attack (float): Time in ms between when the signal rises above threshold and when 
                reduction is applied
            release (float): Time in ms between when the signal fall below threshold and 
                when reduction is decreased.
            makeup (float): Factor of amplification post-processing
            knee (float): Softens the transition between reduction and lack of thereof. 
                Higher values translate to a softer transition. 
            link (str): Choose average between all channels or mean. String of either
                "average" or "mean.
            detection (str): Whether to process exact signal of the RMS of nearby signals. 
                Either "peak" for exact or "rms".
            mix (float): Proportion of compressed signal in output.
            kwargs: Arguments passed to `ffmpeg.filter`
        Returns:
            self: Initial AudioSignal with updated effect chains

        See Also:
            * :func:`apply_effects`: Applies effects once they are in the effects chain.
            * :func:`make_effect`: Syntactic sugar for adding an effect to the chain by name.
            * :func:`reset_effects_chain`: Empties the effects chain without applying any effects.
        """
        fx = effects.compressor(level_in, mode=mode, reduction_ratio=reduction_ratio,
                                attack=attack, release=release, makeup=makeup, knee=knee, link=link,
                                detection=detection, mix=mix, threshold=threshold, **kwargs)
        self._effects_chain.append(fx)
        return self

    def equalizer(self, bands, **kwargs):
        """
        Add eqaulizer effect to AudioSignal's effect chain

        This is a FFmpeg effect. Please see
        https://ffmpeg.org/ffmpeg-all.html#anequalizer  
        for details.

        Notes:
            This effect won't be applied until you call `apply_effect()`!

        Args:
            bands: A list of dictionaries, for each band. The required values for each dictionary:
                'chn': List of channel numbers to apply filter. Must be list of ints.
                'f': central freqency of band
                'w': Width of the band in Hz
                'g': Band gain in dB
                't': Set filter type for band, optional, can be:
                    0, for Butterworth
                    1, for Chebyshev type 1
                    2, for Chebyshev type 2
        Returns:
            self: Initial AudioSignal with updated effect chains

        See Also:
            * :func:`apply_effects`: Applies effects once they are in the effects chain.
            * :func:`make_effect`: Syntactic sugar for adding an effect to the chain by name.
            * :func:`reset_effects_chain`: Empties the effects chain without applying any effects.
        """
        self._effects_chain.append(effects.equalizer(bands, **kwargs))
        return self

    ##################################################
    #              Operator overloading              #
    ##################################################

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.subtract(other)

    def _verify_audio(self, other):
        if self.num_channels != other.num_channels:
            raise AudioSignalException('Cannot do operation with two signals that have '
                                       'a different number of channels!')

        if self.sample_rate != other.sample_rate:
            raise AudioSignalException('Cannot do operation with two signals that have '
                                       'different sample rates!')

    def _verify_audio_arithmetic(self, other):
        self._verify_audio(other)

        if self.signal_length != other.signal_length:
            raise AudioSignalException('Cannot do arithmetic with signals of different length!')

    def __iadd__(self, other):
        return self + other

    def __isub__(self, other):
        return self - other

    def __mul__(self, value):
        if not isinstance(value, numbers.Real):
            raise AudioSignalException('Can only multiply/divide by a scalar!')

        return self.make_copy_with_audio_data(np.multiply(self.audio_data, value), verbose=False)

    def __div__(self, value):
        if not isinstance(value, numbers.Real):
            raise AudioSignalException('Can only multiply/divide by a scalar!')

        return self.make_copy_with_audio_data(np.divide(self.audio_data, float(value)),
                                              verbose=False)

    def __truediv__(self, value):
        return self.__div__(value)

    def __itruediv__(self, value):
        return self.__idiv__(value)

    def __imul__(self, value):
        return self.apply_gain(value)

    def __idiv__(self, value):
        return self.apply_gain(1 / float(value))

    def __len__(self):
        return self.signal_length

    def __eq__(self, other):
        for k, v in list(self.__dict__.items()):
            if isinstance(v, np.ndarray):
                if not np.array_equal(v, other.__dict__[k]):
                    return False
            elif v != other.__dict__[k]:
                return False
        return True

    def __ne__(self, other):
        return not self == other


class AudioSignalException(Exception):
    """
    Exception class for :class:`AudioSignal`.
    """
    pass
