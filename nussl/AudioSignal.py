#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import os.path
import numpy as np
import scipy.io.wavfile as wav

from WindowType import WindowType
import FftUtils
import Constants


class AudioSignal(object):
    """Defines properties of an audio signal and performs basic operations such as Wav loading and STFT/iSTFT.

    Parameters:
        path_to_input_file (string): string specifying path to file. Either this or timeSeries must be provided
        audio_data_array (np.array): Numpy matrix containing a time series of a signal
        signalStartingPosition (Optional[int]): Starting point of the section to be extracted in seconds. Defaults to 0
        signalLength (Optional[int]): Length of the signal to be extracted. Defaults to full length of the signal
        SampleRate (Optional[int]): sampling rate to read audio file at. Defaults to Constants.DEFAULT_SAMPLE_RATE
        stft (Optional[np.array]): Optional pre-coumputed complex spectrogram data.

    Attributes:
        window_type(WindowType): type of window to use in operations on the signal. Defaults to WindowType.DEFAULT
        window_length (int): Length of window in ms. Defaults to 0.06 * SampleRate
        num_fft_bins (int): Number of bins for fft. Defaults to windowLength
        overlap_ratio (float): Ratio of window that overlaps in [0,1). Defaults to 0.5
        complex_spectrogram_data (np.array): complex spectrogram of the data
        power_spectrum_data (np.array): power spectrogram of the data
        Fvec (np.array): frequency vector for stft
        Tvec (np.array): time vector for stft
        sample_rate (int): sampling frequency
  
    Examples:
        * create a new signal object:     ``sig=AudioSignal('sample_audio_file.wav')``
        * compute the spectrogram of the new signal object:   ``sigSpec,sigPow,F,T=sig.STFT()``
        * compute the inverse stft of a spectrogram:          ``sigrec,tvec=sig.iSTFT()``
  
    """

    def __init__(self, path_to_input_file=None, audio_data_array=None, signal_starting_position=0, signal_length=0,
                 sample_rate=Constants.DEFAULT_SAMPLE_RATE, stft=None):

        self.path_to_input_file = path_to_input_file
        self._audioData = None
        self.time = np.array([])
        self.sample_rate = sample_rate

        if (path_to_input_file is not None) and (audio_data_array is not None):
            raise Exception('Cannot initialize AudioSignal object with a path AND an array!')

        if path_to_input_file is not None:
            self.load_audio_from_file(self.path_to_input_file, signal_length, signal_starting_position)
        elif audio_data_array is not None:
            self.load_audio_from_array(audio_data_array, sample_rate)

        # do_STFT properties
        self.complex_spectrogram_data = np.array([]) if stft is None else stft  # complex spectrogram
        self.power_spectrum_data = np.array([])  # power spectrogram
        self.freq_vec = np.array([])  # freq. vector
        self.time_vec = np.array([])  # time vector

        # TODO: put these in a window_attributes object and wrap in a property
        self.window_type = WindowType.DEFAULT
        self.window_length = int(0.06 * self.sample_rate)
        self.num_fft_bins = self.window_length
        self.overlap_ratio = 0.5
        self.overlap_samples = int(np.ceil(self.overlap_ratio * self.window_length))
        self.frequency_max_plot = self.sample_rate / 2

    def __str__(self):
        return 'AudioSignal'

    ##################################################
    # Plotting
    ##################################################

    def plot_time_domain(self):
        raise NotImplementedError('Not ready yet!')

    def plot_freq_domain(self):
        raise NotImplementedError('Not ready yet!')

    ##################################################
    # Properties
    ##################################################

    # Constants for accessing _audio_data np.array indices
    _LEN = 1
    _CHAN = 0

    @property
    def signal_length(self):
        """The length of the audio signal represented by this object in samples
        """
        return self._audioData.shape[self._LEN]

    @property
    def signal_duration(self):
        """The length of the audio signal represented by this object in seconds
        """
        return self.signal_length / self.sample_rate

    @property
    def num_channels(self):
        """The number of channels
        """
        return self._audioData.shape[self._CHAN]

    @property
    def audio_data(self):
        """A numpy array that represents the audio
        """
        return self._audioData

    @audio_data.setter
    def audio_data(self, value):
        assert (type(value) == np.ndarray)

        self._audioData = value

        if self._audioData.ndim < 2:
            self._audioData = np.expand_dims(self._audioData, axis=self._CHAN)

    @property
    def file_name(self):
        """The name of the file wth extension, NOT the full path
        """
        if self.path_to_input_file is not None:
            return os.path.split(self.path_to_input_file)[1]
        return None

    ##################################################
    # I/O
    ##################################################

    def load_audio_from_file(self, input_file_path, signal_starting_position=0, signal_length=0):
        """Loads an audio signal from a .wav file

        Parameters:
            input_file_path: path to input file.
            signal_length (Optional[int]): Length of signal to load. signal_length of 0 means read the whole file
             Defaults to the full length of the signal
            signal_starting_position (Optional[int]): The starting point of the section to be extracted (seconds).
             Defaults to 0 seconds

        """
        try:
            self.sample_rate, audio_input = wav.read(input_file_path)
        except Exception, e:
            print "Cannot read from file, {file}".format(file=input_file_path)
            raise e

        audio_input = audio_input.T

        # Change from fixed point to floating point
        audio_input = audio_input.astype('float') / (np.iinfo(audio_input.dtype).max + 1.0)

        # TODO: the logic here needs work
        if signal_length == 0:
            self.audio_data = np.array(audio_input)
        else:
            start_position = int(signal_starting_position * self.sample_rate)
            self.audio_data = np.array(audio_input[start_position: start_position + self.signal_length, :])

        self.time = np.array((1. / self.sample_rate) * np.arange(self.signal_length))

    def load_audio_from_array(self, signal, sample_rate=Constants.DEFAULT_SAMPLE_RATE):
        """Loads an audio signal from a numpy array

        Parameters:
            signal (np.array): np.array containing the audio file signal sampled at sampleRate
            sample_rate (Optional[int]): the sampleRate of signal. Default is Constants.DEFAULT_SAMPLE_RATE

        """
        self.path_to_input_file = None
        self.audio_data = np.array(signal)
        self.sample_rate = sample_rate

        self.time = np.array((1. / self.sample_rate) * np.arange(self.signal_length))

    def write_audio_to_file(self, output_file_path, sample_rate=None, verbose=False):
        """Outputs the audio signal to a .wav file

        Parameters:
            output_file_path (str): Filename where waveform will be saved
            sample_rate (Optional[int]): The sample rate to write the file at. Default is AudioSignal.SampleRate, which
            is the samplerate of the original signal.
            verbose (Optional[bool]): Flag controlling printing when writing the file.
        """
        if self.audio_data is None:
            raise Exception("Cannot write audio file because there is no audio data.")

        try:
            self.peak_normalize()

            if sample_rate is None:
                sample_rate = self.sample_rate

            wav.write(output_file_path, sample_rate, self.audio_data.T)
        except Exception, e:
            print "Cannot write to file, {file}.".format(file=output_file_path)
            raise e
        if verbose:
            print "Successfully wrote {file}.".format(file=output_file_path)

    ##################################################
    #               STFT Utilities
    ##################################################

    def do_STFT(self):
        """computes the STFT of the audio signal

        Returns:
            * **complex_spectrogram_data** (*np.array*) - complex stft data

            * **power_spectrum_data** (*np.array*) - power spectrogram

            * **Fvec** (*np.array*) - frequency vector

            * **Tvec** (*np.array*) - vector of time frames

        """
        if self.audio_data is None:
            raise Exception("No audio data to make do_STFT from.")

        for i in range(1, self.num_channels + 1):
            Xtemp, Ptemp, Ftemp, Ttemp = FftUtils.f_stft(self.get_channel(i).T, num_ffts=self.num_fft_bins,
                                                         win_length=self.window_length, window_type=self.window_type,
                                                         window_overlap=self.overlap_samples,
                                                         sample_rate=self.sample_rate)

            if np.size(self.complex_spectrogram_data) == 0:
                self.complex_spectrogram_data = Xtemp
                self.power_spectrum_data = Ptemp
                self.freq_vec = Ftemp
                self.time_vec = Ttemp
            else:
                self.complex_spectrogram_data = np.dstack([self.complex_spectrogram_data, Xtemp])
                self.power_spectrum_data = np.dstack([self.power_spectrum_data, Ptemp])

        return self.complex_spectrogram_data, self.power_spectrum_data, self.freq_vec, self.time_vec

    def do_iSTFT(self):
        """Computes and returns the inverse STFT.

        Warning:
            Will overwrite any data in self.AudioData!

        Returns:
             audio_data (np.array): time-domain signal
             time (np.array): time vector
        """
        if self.complex_spectrogram_data.size == 0:
            raise Exception('Cannot do inverse do_STFT without do_STFT data!')

        self.audio_data = np.array([])
        for i in range(1, self.num_channels + 1):
            x_temp, t_temp = FftUtils.f_istft(self.complex_spectrogram_data, self.window_length, self.window_type,
                                              self.overlap_samples, self.sample_rate)

            if np.size(self.audio_data) == 0:
                self.audio_data = np.array(x_temp).T
                self.time = np.array(t_temp).T
            else:
                self.audio_data = np.hstack([self.audio_data, np.array(x_temp).T])

        if len(self.audio_data.shape) == 1:
            self.audio_data = np.expand_dims(self.audio_data, axis=1)

        return self.audio_data, self.time

    ##################################################
    #                  Utilities
    ##################################################

    def concat(self, other):
        """ Add two AudioSignal objects (by adding self.audio_data) temporally.

        Parameters:
            other (AudioSignal): Audio Signal to concatenate with the current one.
        """
        if self.num_channels != other.num_channels:
            raise Exception('Cannot concat two signals that have a different number of channels!')

        self.audio_data = np.concatenate((self.audio_data, other.audio_data))

    def get_channel(self, n):
        """Gets the n-th channel. 1-based.

        Parameters:
            n (int): index of channel to get 1-based.
        Returns:
             n-th channel (np.array): the data in the n-th channel of the signal
        """
        return self.audio_data[n - 1,]

    def peak_normalize(self, bitDepth=16):
        """Normalizes the whole audio file to the max value.

        Parameters:
            bitDepth (Optional[int]): Max value (in bits) to normalize to. Default is 16 bits.
        """
        bitDepth -= 1
        maxVal = 1.0
        maxSignal = np.max(np.abs(self.audio_data))
        if maxSignal > maxVal:
            self.audio_data = np.divide(self.audio_data, maxSignal)

    def add(self, other):
        """adds two audio signals

        Parameters:
            other (AudioSignal): Other audio signal to add.
        Returns:
            sum (AudioSignal): AudioSignal with the sum of the current object and other.
        """
        return self + other

    def sub(self, other):
        """subtracts two audio signals

        Parameters:
            other (AudioSignal): Other audio signal to subtract.
        Returns:
            diff (AudioSignal): AudioSignal with the difference of the current object and other.
        """
        return self - other

    ##################################################
    #              Operator overloading
    ##################################################

    def __add__(self, other):
        if self.num_channels != other.num_channels:
            raise Exception('Cannot add two signals that have a different number of channels!')

        # for ch in range(self.num_channels):
        # TODO: make this work for multiple channels
        if self.audio_data.size > other.audio_data.size:
            combined = np.copy(self.audio_data)
            combined[0: other.audio_data.size] += other.audio_data
        else:
            combined = np.copy(other.audio_data)
            combined[0: self.audio_data.size] += self.audio_data

        return AudioSignal(audio_data_array=combined)

    def __sub__(self, other):
        if self.num_channels != other.num_channels:
            raise Exception('Cannot subtract two signals that have a different number of channels!')

        # for ch in range(self.num_channels):
        # TODO: make this work for multiple channels
        if self.audio_data.size > other.audio_data.size:
            combined = np.copy(self.audio_data)
            combined[0: other.audio_data.size] -= other.audio_data
        else:
            combined = np.copy(other.audio_data)
            combined[0: self.audio_data.size] -= self.audio_data

        return AudioSignal(audio_data_array=combined)

    def __iadd__(self, other):
        return self + other

    def __isub__(self, other):
        return self - other

    def __len__(self):
        return len(self.audio_data)
