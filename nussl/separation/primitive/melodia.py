import numpy as np
from scipy.ndimage.filters import convolve
from scipy.ndimage import maximum_filter, gaussian_filter
import scipy.special

from .. import MaskSeparationBase, SeparationException
from ..benchmark import HighLowPassFilter
from ... import AudioSignal
from ... import vamp_imported
import norbert

if vamp_imported:
    import vamp

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
        input_audio_signal: (AudioSignal object) The AudioSignal object that has the
          audio data that Melodia will be run on.

        high_pass_cutoff: (Optional) (float) value (in Hz) for the high pass cutoff 
          filter.
        
        minimum_frequency: (float) minimum frequency in Hertz (default 55.0)

        maximum_frequency: (float) maximum frequency in Hertz (default 1760.0)

        voicing_tolerance: (float) Greater values will result in more pitch contours 
          included in the final melody. Smaller values will result in less pitch 
          contours included in the final melody (default 0.2).

        minimum_peak_salience: (float) a hack to avoid silence turning into junk 
          contours when analyzing monophonic recordings (e.g. solo voice with 
          no accompaniment). Generally you want to leave this untouched (default 0.0).

        num_overtones: (Optional) (int) Number of overtones to use when creating 
          melody mask.
    """
    def __init__(self, input_audio_signal, high_pass_cutoff=100, minimum_frequency=55.0,
                 maximum_frequency=1760.0, voicing_tolerance=0.2, minimum_peak_salience=0.0,
                 num_overtones=120, smooth_length=5, mask_type='soft', mask_threshold=0.5):
    
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
        hop = 128./44100. # hard coded hop in Melodia vamp plugin, converting it to frames.
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
        transition_length = .001 # duration for fade in/out and frequency interpretation
        phase = np.zeros(num_overtones)
        previous_frequency = 0
        previous_time = 0

        for time, frequency in zip(self.timestamps, self.melody):
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
                    weight = np.exp(-overtone)
                    samples += weight * np.sin(phases)
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
        melody_signal /= float(max(np.max(melody_signal), 1e-7))
        melody_signal = [melody_signal for channel in range(self.audio_signal.num_channels)]
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
        stft = stft ** 2
        stft /= np.maximum(np.max(stft, axis=1, keepdims=True), 1e-7)

        mask = np.empty(self.stft.shape)

        # Smoothing the mask row-wise using a low-pass filter to
        # get rid of discontuinities in the mask.
        kernel =  np.full((1, self.smooth_length), 1/self.smooth_length)
        for ch in range(self.audio_signal.num_channels):
            mask[..., ch] = convolve(stft[..., ch], kernel)
            mask[..., ch] = maximum_filter(mask[..., ch], size=(5, 1))
            mask[..., ch] = gaussian_filter(mask[..., ch], sigma=1)
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

        if self.mask_type == self.MASKS['binary']:
            foreground_mask = foreground_mask.mask_to_binary(
                self.mask_threshold)

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
