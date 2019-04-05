from torch.utils.data import Dataset
import pickle
import librosa
import numpy as np
import os
import shutil
import random
from typing import Dict, Any, Optional, Tuple, List
from ...core import AudioSignal

class BaseDataset(Dataset):
    def __init__(self, folder: str, options: Dict[str, Any]):
        """Implements a variety of methods for loading source separation
        datasets such as WSJ0-[2,3]mix and datasets made with Scaper.

        Arguments:
            folder - Folder where dataset is contained.

        Keyword Arguments:
            options - a dictionary containing the settings for the dataset
                loader. See `config/defaults/metadata/dataset.json` for full
                description.
        """

        self.folder = folder
        self.files = self.get_files(self.folder)
        self.cached_files = []
        self.options = options
        self.targets = [
            'log_spectrogram',
            'magnitude_spectrogram',
            'assignments',
            'source_spectrograms',
            'weights'
        ]
        self.data_keys_for_training = []
        self.cache_input = (
            self.options['cache']
            if self.options['cache']
            else None
        )

        self.create_cache_folder()

        if self.options['fraction_of_dataset'] < 1.0:
            num_files = int(
                len(self.files) * self.options['fraction_of_dataset']
            )
            random.shuffle(self.files)
            self.files = self.files[:num_files]

    def create_cache_folder(self):
        if self.options['cache']:
            self.cache = os.path.join(
                os.path.expanduser(self.options['cache']),
                '_'.join(self.folder.split('/')),
                self.options['output_type'],
                '_'.join(self.options['weight_type'])
            )
            print(f'Caching to: {self.cache}')
            os.makedirs(self.cache, exist_ok=True)


    def get_files(self, folder):
        raise NotImplementedError()

    def load_audio_files(
            self,
            filename: str
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Loads audio file with given name

        Path to find given filename at determined by implementation by subclass

        Args:
            filename - name of file to load

        Returns:
            TODO: flesh this out
            tuple of mix (np.ndarray? - shape?), sources (np.ndarray? - shape?),
            assignments? (np.ndarray? - shape?)
        """
        raise NotImplementedError()

    def __len__(self) -> int:
        """Gets number of examples"""
        return len(self.files)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        """Gets one item from dataset

        Args:
            i - index of example to get

        Returns:
            one data point (an output dictionary containing the data comprising
            one example)
        """
        return self._get_item_helper(self.files[i], self.cache, i)

    def clear_cache(self):
        print(f'Clearing cache: {self.cache}')
        shutil.rmtree(self.cache, ignore_errors=True)
    
    def populate_cache(self, filename, i):
        output = self._generate_example(filename)
        if self.data_keys_for_training:
            output = [
                {k: o[k] for k in o if k in self.data_keys_for_training}
                for o in output
            ]
        for j, o in enumerate(output):
            _filepart = f'{i:08d}.pth.part{j}'
            self.write_to_cache(o, _filepart)
        return random.choice(output)

    def switch_to_cache(self):
        self.original_files = self.files
        self.files = [x for x in os.listdir(self.cache) if '.part' in x]
        random.shuffle(self.files)

    def _get_item_helper(
        self,
        filename: str,
        cache: Optional[str],
        i: int = -1,
    ) -> Dict[str, Any]:
        """Gets one item from dataset

        If `cache` is None, will generate an example (training|validation) from
        scratch. If `cache` is not None, it will attempt to read from the path
        given by `cache`. On failure it will write to the path given by `cache`
        for subsequent reads.

        Args:
            filename - name of file corresponding to current example
            cache - `None` or path to cache folder
            i - index of current example (used only in cache filename
                generation). Defaults to -1 (should only be `-1` when `cache` is
                `None`)

        Returns:
            one data point (an output dictionary containing the data comprising
            one example)
        """
        if self.cache:
            try:
                return self.load_from_cache(filename)
            except:
                return self.populate_cache(filename, i)
        else:
            return random.choice(self._generate_example(filename))

    def _generate_example(self, filename: str) -> List[Dict[str, Any]]:
        """Generates one example (training|validation) from given filename

        Args:
            filename - name of audio file from which to generate example

        Returns:
            one data point (an output dictionary containing the data comprising
            one example)
        """
        mix, sources, classes = self.load_audio_files(filename)
        output = self.construct_input_output(mix, sources)
        output['weights'] = self.get_weights(
            output,
            self.options['weight_type']
        )
        output['log_spectrogram'] = self.whiten(output['log_spectrogram'])
        output['classes'] = classes
        output = self.get_target_length_and_transpose(
            output,
            self.options['length']
        )

        return [self.format_output(o) for o in output]

    def format_output(self, output):
        # [num_batch, sequence_length, num_frequencies*num_channels, ...]
        for key in self.targets:
            if self.options['format'] == 'rnn':
                _shape = output[key].shape
                shape = [_shape[0], _shape[1], _shape[2]]
                if len(_shape) > 3:
                    shape += _shape[3:]
                output[key] = np.reshape(output[key], shape)
            elif self.options['format'] == 'cnn':
                axes_loc = [0, 3, 2, 1]
                output[key] = np.moveaxis(output[key], [0, 1, 2, 3], axes_loc)

        return output

    def write_to_cache(self, data_dict, filename):
        with open(os.path.join(self.cache, filename), 'wb') as f:
            pickle.dump(data_dict, f)

    def load_from_cache(self, filename: str):
        with open(os.path.join(self.cache, filename), 'rb') as f:
            data = pickle.load(f)
        return data

    def whiten(self, data):
        data -= data.mean()
        data /= (data.std() + 1e-7)
        return data

    def construct_input_output(self, mix, sources):
        log_spectrogram, mix_stft = self.transform(mix)
        mix_magnitude, mix_phase = np.abs(mix_stft), np.angle(mix_stft)
        source_magnitudes = []
        source_log_magnitudes = []

        for source in sources:
            source_log_magnitude, source_stft = self.transform(source)
            source_magnitude, source_phase = (
                np.abs(source_stft),
                np.angle(source_stft)
            )
            if self.options['output_type'] == 'msa':
                source_magnitude = np.minimum(mix_magnitude, source_magnitude)
            elif self.options['output_type'] == 'psa':
                source_magnitude = np.maximum(
                    0.0,
                    np.minimum(
                        mix_magnitude,
                        source_magnitude * np.cos(source_phase - mix_phase),
                    )
                )
            source_magnitudes.append(source_magnitude)
            source_log_magnitudes.append(source_log_magnitude)

        source_magnitudes = np.stack(source_magnitudes, axis=-1)
        source_log_magnitudes = np.stack(source_log_magnitudes, axis=-1)

        shape = source_magnitudes.shape
        source_log_magnitudes = source_log_magnitudes.reshape(
            np.prod(shape[0:-1]),
            shape[-1],
        )

        assignments = np.zeros(source_log_magnitudes.shape)
        source_argmax = np.argmax(source_log_magnitudes, axis=-1)
        assignments[np.arange(assignments.shape[0]), source_argmax] = 1.0
        assignments = assignments.reshape(shape)

        output = {
            'log_spectrogram': log_spectrogram,
            'magnitude_spectrogram': mix_magnitude,
            'assignments': assignments,
            'source_spectrograms': source_magnitudes,
        }

        return output


    def get_target_length_and_transpose(self, data_dict, target_length):
        length = data_dict['log_spectrogram'].shape[1]

        # Break up data into sequences of target length.  Return a list.
        offsets = np.arange(0, length,  target_length)
        offsets[-1] = max(0, length - target_length)

        # Select random offset, return that.
        offsets = [np.random.randint(0, max(0, length - target_length))]
        output_data_dicts = []
    
        for i, target in enumerate(self.targets):
            data = data_dict[target]
            pad_length = max(target_length - length, 0)
            pad_tuple = [(0, 0) for k in range(len(data.shape))]
            pad_tuple[1] = (0, pad_length)
            data_dict[target] = np.pad(data, pad_tuple, mode='constant')

        for offset in offsets:
            _data_dict = data_dict.copy()
            for target in self.targets:
                _data_dict[target] = _data_dict[target][
                    :,
                    offset:offset + target_length,
                    :self.options['num_channels']
                ]
                _data_dict[target] = np.swapaxes(_data_dict[target], 0, 1)
            output_data_dicts.append(_data_dict)

        return output_data_dicts

    @staticmethod
    def transform(audio_signal):
        """Uses nussl STFT to transform.

        Arguments:
            audio_signal {[np.ndarray]} -- AudioSignal object

        Returns:
            [tuple] -- (log_spec, stft, n). log_spec contains the
            log_spectrogram, stft contains the complex spectrogram, and n is the
        """
        stft = (
            audio_signal.stft() 
            if audio_signal.stft_data is None 
            else audio_signal.stft_data
        )
        log_spectrogram = librosa.amplitude_to_db(np.abs(stft))
        return log_spectrogram, stft


    def get_weights(self, data_dict, weight_type):
        weights = np.ones(data_dict['magnitude_spectrogram'].shape)
        if ('magnitude' in weight_type):
            weights *= self.magnitude_weights(
                data_dict['magnitude_spectrogram']
            )
        if ('threshold' in weight_type):
            weights *= self.threshold_weights(
                data_dict['log_spectrogram'],
                self.options['weight_threshold']
            )
        return weights

    @staticmethod
    def magnitude_weights(magnitude_spectrogram):
        weights = magnitude_spectrogram / (np.sum(magnitude_spectrogram) + 1e-6)
        weights *= (
            magnitude_spectrogram.shape[0] * magnitude_spectrogram.shape[1]
        )
        return weights

    @staticmethod
    def threshold_weights(log_spectrogram, threshold=-40):
        return (
            (log_spectrogram - np.max(log_spectrogram)) > threshold
        ).astype(np.float32)

    def _load_audio_file(self, file_path: str) -> AudioSignal:
        """Loads audio file at given path. Uses 
        Args:
            file_path - relative or absolute path to file to load

        Returns:
            tuple of loaded audio w/ shape ? and sample rate
        """
        audio_signal = AudioSignal(file_path, sample_rate=self.options['sample_rate'])
        audio_signal.stft_params.window_length = self.options['n_fft']
        audio_signal.stft_params.hop_length = self.options['hop_length']
        return audio_signal