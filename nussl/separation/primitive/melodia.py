import numpy as np
from scipy.ndimage.filters import convolve
from scipy.ndimage import maximum_filter, gaussian_filter

from .. import MaskSeparationBase, SeparationException
from ..benchmark import HighLowPassFilter
from ... import AudioSignal
from ... import vamp_imported
import numpy as np
import scipy.signal

if vamp_imported:
    import vamp

# function for generating the vocal chord impulse response
def rosenmodel(t1, t2, fs):
    """
    This model for generating singing vowel sounds from sine tones comes
    from:

    https://simonl02.users.greyc.fr/files/Documents/Teaching/SignalProcessingLabs/lab3.pdf

    The above will be referred to throughout these docstrings as THE DOCUMENT.

    Original author: Fatemeh Pishdadian

    The arguments to the fucntions throughout this text follow the 
    signatures laid out in THE DOCUMENT.

    This is used in Melodia to generate the melody signal to produce a 
    mask.

    Equation 2 in THE DOCUMENT.
    """

    N1 = np.floor(t1 * fs)
    N2 = np.floor(t2 * fs)

    samp_vec1 = np.arange(N1+1)
    samp_vec2 = np.arange(N1,N1+N2+1)

    ir_func1 =  0.5 * (1 - np.cos((np.pi * samp_vec1)/N1))
    ir_func2 = np.cos(np.pi * (samp_vec2 - N1)/(2 * N2))

    vchord_filt = np.concatenate((ir_func1,ir_func2))

    return vchord_filt


# function for computing the denominator coeffs of the vocal cavity filter transfer function
def oral_cavity_filt(pole_amps, pole_freqs,fs):
    """
    This model for generating singing vowel sounds from sine tones comes
    from:

    https://simonl02.users.greyc.fr/files/Documents/Teaching/SignalProcessingLabs/lab3.pdf

    The above will be referred to throughout these docstrings as THE DOCUMENT.

    Original author: Fatemeh Pishdadian

    The arguments to the fucntions throughout this text follow the 
    signatures laid out in THE DOCUMENT.

    This is used in Melodia to generate the melody signal to produce a 
    mask.

    Solves "Q. Write a function to synthesize filter H(z)" in 
    THE DOCUMENT
    """
    num_pole_pair = len(pole_amps)
    poles = pole_amps * np.exp(1j * 2 * np.pi * pole_freqs / fs)
    poles_conj = np.conj(poles)

    denom_coeffs = 1
    for i in range(num_pole_pair):
        pole_temp = poles[i]
        pole_conj_temp = poles_conj[i]
        pole_pair_coeffs = np.convolve(np.array([1,-pole_temp]),np.array([1,-pole_conj_temp]))

        denom_coeffs = np.convolve(denom_coeffs, pole_pair_coeffs)
    return denom_coeffs

def _apply_vowel_filter(impulse_train, fs, t1=0.0075, t2=.013,
                     pole_amps=None, pole_freqs=None):
    """
    This model for generating singing vowel sounds from sine tones comes
    from:

    https://simonl02.users.greyc.fr/files/Documents/Teaching/SignalProcessingLabs/lab3.pdf

    The above will be referred to throughout these docstrings as THE DOCUMENT.

    Original author: Fatemeh Pishdadian

    The arguments to the fucntions throughout this text follow the 
    signatures laid out in THE DOCUMENT.

    This is used in Melodia to generate the melody signal to produce a 
    mask.
    
    Args:
        impulse_train (np.ndarray): Numpy array with data to be filtered
        fs (int): Sample rate of audio.
        t1 (float, optional): N1 in Equation 2 in THE DOCUMENT. Defaults to 0.0075.
        t2 (float, optional): N2 in Equation 2 in THE DOCUMENT. Defaults to .013.
        pole_amps (np.ndarray, optional): Pole amplitudes, see Figures 2-4 in THE DOCUMENT. 
          Defaults to None, which maps to E vowel.
        pole_freqs (np.ndarray, optional): Pole frequencies, see Figures 2-4 in THE DOCUMENT. 
          Defaults to None, which maps to E vowel
    
    Returns:
        np.ndarray: Filtered impulse train that should sound sort of like the desired 
          vowel.
    """
    if pole_amps is None:
        pole_amps = np.array([0.99,0.98,0.9,0.9])
    if pole_freqs is None:
        pole_freqs = np.array([800,1200,2800,3600])
        
    vchord_filt = rosenmodel(t1, t2, fs)
    vchord_out = np.convolve(impulse_train, vchord_filt)
        
    denom_coeffs = oral_cavity_filt(pole_amps, pole_freqs, fs)
    oral_out = scipy.signal.lfilter(
        np.array([1]), denom_coeffs, vchord_out)
    lip_out = np.real(scipy.signal.lfilter(
        np.array([1,-1]), np.array([1]), oral_out))
    
    lip_out = lip_out[:impulse_train.shape[0]]
    
    return np.real(lip_out)

class Melodia(MaskSeparationBase):
    """
    Implements melody extraction using Melodia [1].

    This needs Melodia installed as a vamp plugin, as well as having vampy for 
    Python installed. Install Melodia via: https://www.upf.edu/web/mtg/melodia.
    Note that Melodia can be used only for NON-COMMERCIAL use.

    References:

    [1] J. Salamon and E. Gómez, "Melody Extraction from Polyphonic Music Signals using 
        Pitch Contour Characteristics", IEEE Transactions on Audio, Speech and 
        Language Processing, 20(6):1759-1770, Aug. 2012.

    Args:
        input_audio_signal (AudioSignal object): The AudioSignal object that has the
          audio data that Melodia will be run on.

        high_pass_cutoff (optional, float): value (in Hz) for the high pass cutoff 
          filter.
        
        minimum_frequency (optional, float): minimum frequency in Hertz (default 55.0)

        maximum_frequency (optional, float): maximum frequency in Hertz (default 1760.0)

        voicing_tolerance (optional, float): Greater values will result in more pitch contours 
          included in the final melody. Smaller values will result in less pitch 
          contours included in the final melody (default 0.2).

        minimum_peak_salience (optional, float): a hack to avoid silence turning into junk 
          contours when analyzing monophonic recordings (e.g. solo voice with 
          no accompaniment). Generally you want to leave this untouched (default 0.0).

        num_overtones (optional, int): Number of overtones to use when creating 
          melody mask.

        apply_vowel_filter (optional, bool): Whether or not to apply a vowel filter
          on the resynthesized melody signal when masking.

        smooth_length (optional, int): Number of frames to smooth discontinuities in the
          mask.

        add_lower_octave (optional, fool): Use octave below fundamental frequency as well
          to take care of octave errors in pitch tracking, since we only care about
          the mask. Defaults to False.
        
        mask_type (optional, str): Type of mask to use.

        mask_threshold (optional, float): Threshold for mask to convert to binary.
    """

    def __init__(self, input_audio_signal, high_pass_cutoff=100, minimum_frequency=55.0,
                 maximum_frequency=1760.0, voicing_tolerance=0.2, minimum_peak_salience=0.0,
                 compression=0.5, num_overtones=40, apply_vowel_filter=False, smooth_length=5, 
                 add_lower_octave=False, mask_type='soft', mask_threshold=0.5):
        # lazy load vamp to check if it exists
        from ... import vamp_imported

        melodia_installed = False
        if vamp_imported:
            melodia_installed = 'mtg-melodia:melodia' in vamp.list_plugins()

        if not vamp_imported or not melodia_installed:
            self._raise_vamp_melodia_error()

        super().__init__(
            input_audio_signal=input_audio_signal,
            mask_type=mask_type,
            mask_threshold=mask_threshold
        )

        self.high_pass_cutoff = high_pass_cutoff
        self.minimum_frequency = float(minimum_frequency)
        self.maximum_frequency = float(maximum_frequency)
        self.voicing_tolerance = float(voicing_tolerance)
        self.minimum_peak_salience = float(minimum_peak_salience)
        self.compression = compression
        self.apply_vowel_filter = apply_vowel_filter
        self.add_lower_octave = add_lower_octave

        self.melody = None
        self.melody_signal = None
        self.timestamps = None

        self.num_overtones = num_overtones
        self.smooth_length = smooth_length

    def _raise_vamp_melodia_error(self):
        raise SeparationException(
            '\n**~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~**'
            '\n*          Are Vamp and Melodia installed correctly?          *'
            '\n* Check https://bit.ly/2DXbrAk for installation instructions! *'
            '\n**~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~**')

    def extract_melody(self):
        """
        Extracts melody from the audio using the melodia vamp plugin. Uses arguments kept 
        in self:
        
        - `self.minimum_frequency` (default: 55 Hz)
        - `self.maximum_frequency` (default: 1760 Hz)
        - `self.voicing_tolerance` (default: 0.2)
        - `self.minimum_peak_salience` (default: 0.0)

        This function sets two class members used in other parts:

        - `self.melody`: (numpy array) contains the melody in Hz for every timestep 
          (0 indicates no voice).
        - `self.timestamps`: (numpy array) contains the timestamps for each melody note
        """

        params = {
            'minfqr': self.minimum_frequency,
            'maxfqr': self.maximum_frequency,
            'voicing': self.voicing_tolerance,
            'minpeaksalience': self.minimum_peak_salience
        }

        data = vamp.collect(self.audio_signal.audio_data, self.sample_rate,
                            "mtg-melodia:melodia", parameters=params)

        _, melody = data['vector']
        hop = 128. / 44100.  # hard coded hop in Melodia vamp plugin, converting it to frames.
        timestamps = 8 * hop + np.arange(len(melody)) * hop
        melody[melody < 0] = 0
        self.melody = melody
        self.timestamps = timestamps

    def create_melody_signal(self, num_overtones):
        """
        Adapted from Melosynth by Justin Salamon: https://github.com/justinsalamon/melosynth. 
        To mask the mixture, we need to identify time-frequency bins that belong to the 
        melody. Melodia outputs only the fundamental frequency of the melodic line. 
        To construct the mask we take the fundamental frequency and add all the 
        overtones of it (up to num_overtones) to the mask. The melody is faded in and 
        out at onsets and offsets to make the separation sound more natural 
        (hard-coded by transition_length).

        Args:

            num_overtones (int): Number of overtones to expand out to build the mask.

        """

        if self.timestamps[0] > 0:
            estimated_hop = np.median(np.diff(self.timestamps))
            previous_time = max(self.timestamps[0] - estimated_hop, 0)
            self.timestamps = np.insert(self.timestamps, 0, previous_time)
            self.melody = np.insert(self.melody, 0, 0)

        sample_rate = self.audio_signal.sample_rate
        melody_signal = []
        transition_length = .001  # duration for fade in/out and frequency interpretation
        phase = np.zeros(num_overtones)
        previous_frequency = 0
        previous_time = 0

        overtone_weights = np.ones(num_overtones)

        for time, frequency in zip(self.timestamps, self.melody):
            if self.add_lower_octave:
                frequency = frequency / 2 
            # taking care of octave errors since we only care about masking
            num_samples = int(np.round((time - previous_time) * sample_rate))
            if num_samples > 0:
                num_transition_samples = float(
                    min(np.round(transition_length * sample_rate), num_samples))
                frequency_series = np.ones(num_samples) * previous_frequency

                if previous_frequency > 0 and frequency > 0:
                    frequency_series += np.minimum(
                        np.arange(num_samples) / num_transition_samples, 1) * \
                                        (frequency - previous_frequency)
                elif frequency > 0:
                    frequency_series = np.ones(num_samples) * frequency

                samples = np.zeros(num_samples)

                for overtone in range(num_overtones):
                    overtone_num = overtone + 1
                    phasors = 2 * np.pi * overtone_num * frequency_series / float(sample_rate)
                    phases = phase[overtone] + np.cumsum(phasors)
                    samples += overtone_weights[overtone] * np.sign(np.sin(phases))
                    phase[overtone] = phases[-1]

                if previous_frequency == 0 and frequency > 0:
                    samples *= np.minimum(np.arange(num_samples) / num_transition_samples, 1)
                elif previous_frequency > 0 and frequency == 0:
                    samples *= np.maximum(1 - np.arange(num_samples) / num_transition_samples, 0)
                elif previous_frequency == 0 and frequency == 0:
                    samples *= 0

                melody_signal.extend(samples)

            previous_frequency = frequency
            previous_time = time

        melody_signal = np.asarray(melody_signal)
        if self.apply_vowel_filter:
            melody_signal = _apply_vowel_filter(melody_signal, sample_rate)

        melody_signal /= float(max(np.max(melody_signal), 1e-7))
        melody_signal = [melody_signal for _ in range(self.audio_signal.num_channels)]
        melody_signal = np.asarray(melody_signal)
        melody_signal = melody_signal[:, 0:self.audio_signal.signal_length]
        melody_signal = AudioSignal(
            audio_data_array=melody_signal,
            sample_rate=sample_rate,
            stft_params=self.audio_signal.stft_params
        )

        self.melody_signal = melody_signal
        return melody_signal

    def create_harmonic_mask(self, melody_signal):
        """
        Creates a harmonic mask from the melody signal. The mask is smoothed to reduce 
        the effects of discontinuities in the melody synthesizer.
        """
        stft = np.abs(melody_signal.stft())

        # Need to threshold the melody stft since the synthesized
        # F0 sequence overtones are at different weights.
        stft = stft ** self.compression
        stft /= np.maximum(np.max(stft, axis=1, keepdims=True), 1e-7)

        mask = np.empty(self.stft.shape)

        # Smoothing the mask row-wise using a low-pass filter to
        # get rid of discontuinities in the mask.
        kernel = np.full((1, self.smooth_length), 1 / self.smooth_length)
        for ch in range(self.audio_signal.num_channels):
            mask[..., ch] = convolve(stft[..., ch], kernel)
        return mask

    def run(self):
        high_low = HighLowPassFilter(self.audio_signal, self.high_pass_cutoff)
        high_pass_masks = high_low.run()

        # separate the mixture foreground melody by masking
        if self.melody_signal is None:
            self.extract_melody()
            self.create_melody_signal(self.num_overtones)

        foreground_mask = self.create_harmonic_mask(self.melody_signal)
        foreground_mask = self.MASKS['soft'](foreground_mask)

        foreground_mask = foreground_mask
        background_mask = foreground_mask.invert_mask()

        _masks = np.stack(
            [background_mask.mask, foreground_mask.mask], axis=-1)

        self.result_masks = []

        for i in range(_masks.shape[-1]):
            mask_data = _masks[..., i]
            if self.mask_type == self.MASKS['binary']:
                mask_data = _masks[..., i] == np.max(_masks, axis=-1)

            if i == 0:
                mask_data = np.maximum(mask_data, high_pass_masks[i].mask)
            elif i == 1:
                mask_data = np.minimum(mask_data, high_pass_masks[i].mask)

            mask = self.mask_type(mask_data)
            self.result_masks.append(mask)

        return self.result_masks
