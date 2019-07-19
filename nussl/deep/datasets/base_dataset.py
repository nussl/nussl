from torch.utils.data import Dataset
import pickle
import librosa
import numpy as np
import os
import shutil
import random
from typing import Dict, Any, Optional, Tuple, List
from ...core import AudioSignal
from scipy.io import wavfile
import logging
import copy
from enum import Enum
import zarr
import numcodecs

class PRECISION(Enum):
    FLOAT32=np.float32
    FLOAT16=np.float16

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
        self.options = copy.deepcopy(options)
        self.use_librosa = self.options.pop('use_librosa_stft', False)
        self.dataset_tag = self.options.pop('dataset_tag', 'default')
        self.current_length = self.options.pop('min_length', self.options['length'])
        self.excerpt_selection_strategy = self.options.pop('excerpt_selection_strategy', 'random')
        self.save_precision = self.options.pop('save_precision', 'FLOAT32')
        self.load_precision = self.options.pop('save_precision', 'FLOAT32')
        self.chunk_size = self.options.pop('chunk_size', 1)

        self.files = self.get_files(self.folder)
        random.shuffle(self.files)
        self.cached_files = []
        self.targets = [
            'log_spectrogram',
            'magnitude_spectrogram',
            'assignments',
            'source_spectrograms',
            'weights'
        ]
        self.data_keys_for_training = self.options.pop('data_keys_for_training', [])
        self.setup_cache()
        self.cache_populated = False

        if self.options['fraction_of_dataset'] < 1.0:
            num_files = int(
                len(self.files) * self.options['fraction_of_dataset']
            )
            self.files = self.files[:num_files]

    def setup_cache(self):
        if self.options['cache']:
            cache = os.path.join(
                os.path.expanduser(self.options['cache']),
                '_'.join(self.folder.split('/')),
                self.options['output_type'],
                '_'.join(self.options['weight_type']),
            )
            self.cache = os.path.join(cache, self.dataset_tag + '.zarr')
            overwrite = self.options.pop('overwrite_cache', False)
            logging.info(f'Caching to: {self.cache}')

            file_mode = 'r'
            if os.path.exists(self.cache):
                logging.info(f'Cache location {self.cache} exists! Checking if overwrite. Otherwise, will use cache.')
                if overwrite:
                    logging.info('Overwriting cache.')
                    file_mode = 'w'
            else:
                logging.info(f'{self.cache} does not exist...creating a new cache')
                file_mode = 'w'
            
            self.cache_dataset = zarr.open(
                self.cache, 
                mode=file_mode, 
                shape=(len(self.files),), 
                chunks=(self.chunk_size,),
                dtype=object, 
                object_codec=numcodecs.Pickle(),
                synchronizer=zarr.ThreadSynchronizer(),
            )

    def clear_cache(self):
        logging.info(f'Clearing cache: {self.cache}')
        shutil.rmtree(self.cache, ignore_errors=True)
    
    def populate_cache(self, filename, i):
        output = self._generate_example(filename)
        if self.data_keys_for_training:
            output = {
                k: output[k] for k in output 
                if k in self.data_keys_for_training
            }
        self.write_to_cache(output, i)
        return self.get_target_length(output, self.current_length)

    def switch_to_cache(self):
        self.cache_dataset = zarr.open(
            self.cache, 
            mode='r', 
            shape=(len(self.files),), 
            chunks=(self.chunk_size,),
            dtype=object, 
            object_codec=numcodecs.Pickle(),
            synchronizer=zarr.ThreadSynchronizer(),
        )
        self.cache_populated = True

    def write_to_cache(self, data_dict, i):
        self.cache_dataset[i] = data_dict

    def load_from_cache(self, i):
        data = self.cache_dataset[i]
        return self.get_target_length(data, self.current_length)
            
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
            if self.cache_populated:
                output = self.load_from_cache(i)
            else:
                output = self.populate_cache(filename, i)
        else:
            output = self._generate_example(filename)
            output = self.get_target_length(output, self.current_length)
            
        return output

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
        output = self.transpose_pad_and_filter(
            output,
            self.options['length']
        )
        return output

    def whiten(self, data):
        return data

    def construct_input_output(self, mix, sources):
        log_spectrogram, mix_stft = self.transform(mix)
        mix_magnitude = np.abs(mix_stft)
        source_magnitudes = []

        for source in sources:
            _, source_stft = self.transform(source)
            source_magnitude = np.abs(source_stft)

            if self.options['output_type'] == 'msa':
                source_magnitude = np.minimum(mix_magnitude, source_magnitude)
            elif self.options['output_type'] == 'psa':
                mix_phase = np.angle(mix_stft)
                source_phase = np.angle(source_stft)
                source_magnitude = np.maximum(
                    0.0,
                    np.minimum(
                        mix_magnitude,
                        source_magnitude * np.cos(source_phase - mix_phase),
                    )
                )
            source_magnitudes.append(source_magnitude)

        source_magnitudes = np.stack(source_magnitudes, axis=-1)
        shape = source_magnitudes.shape
        assignments = (
            source_magnitudes == np.amax(source_magnitudes, axis=-1, keepdims=True)
        ).astype(float)

        output = {
            'log_spectrogram': log_spectrogram,
            'magnitude_spectrogram': mix_magnitude,
            'assignments': assignments,
            'source_spectrograms': source_magnitudes,
        }

        return output
        
    def transpose_pad_and_filter(self, data_dict, max_length):
        length = data_dict['log_spectrogram'].shape[1]
    
        for i, target in enumerate(self.targets):
            data = data_dict[target]
            pad_length = max(max_length - length, 0)
            pad_tuple = [(0, 0) for k in range(len(data.shape))]
            pad_tuple[1] = (0, pad_length)
            data_dict[target] = np.pad(data, pad_tuple, mode='constant')

        for target in self.targets:
            if self.data_keys_for_training and target not in self.data_keys_for_training:
                data_dict.pop(target)
            else:
                data_dict[target] = np.swapaxes(data_dict[target], 0, 1)
        
        return data_dict
    
    def set_current_length(self, current_length):
        if current_length > self.options['length']:
            logging.warning(
                f"current_length={current_length} exceeds original "
                f"set max length {self.options['length']}. "
                f"Setting current_length to {self.options['length']}")
            current_length = self.options['length']
        self.current_length = current_length

    def get_target_length(self, data_dict, target_length):
        length = data_dict['log_spectrogram'].shape[0]
        if self.excerpt_selection_strategy == 'random':
            offset = np.random.randint(0, max(1, length - target_length))
        elif self.excerpt_selection_strategy == 'balanced':
            _balance = data_dict['assignments'].mean(axis=-3).prod(axis=-1)
            indices = np.argwhere(_balance >= np.percentile(_balance, 50))[:, 0]
            indices[indices > length - target_length] = max(0, length - target_length)
            indices = np.unique(indices)
            offset = np.random.choice(indices)
                
        for target in data_dict:
            if target != 'classes':
                data_dict[target] = data_dict[target][
                    offset:offset + target_length,
                    :,
                    :self.options['num_channels']
                ]
        return data_dict

    def transform(self, audio_signal):
        """Uses nussl STFT to transform.

        Arguments:
            audio_signal {[np.ndarray]} -- AudioSignal object

        Returns:
            [tuple] -- (log_spec, stft, n). log_spec contains the
            log_spectrogram, stft contains the complex spectrogram, and n is the
        """
        audio_signal.stft_data = None
        stft = (
            audio_signal.stft(use_librosa=self.use_librosa)
        )
        log_spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        return log_spectrogram, stft


    def get_weights(self, data_dict, weight_type):
        weights = np.ones(data_dict['magnitude_spectrogram'].shape)
        if ('magnitude' in weight_type):
            weights *= self.magnitude_weights(
                data_dict['magnitude_spectrogram']
            )
        elif ('source_magnitude' in weight_type):
            weights *= self.source_magnitude_weights(
                data_dict['source_spectrograms']
            )
        if ('threshold' in weight_type):
            weights *= self.threshold_weights(
                data_dict['log_spectrogram'],
                self.options['weight_threshold']
            )
        if ('class' in weight_type):
            weights *= self.class_weights(
                data_dict['assignments'],
            )
        if ('log' in weight_type):
            weights = np.log10(weights + 1)
        return np.sqrt(weights)

    @staticmethod
    def class_weights(assignments):
        _shape = assignments.shape 
        assignments = assignments.reshape(-1, _shape[-1])

        class_weights = assignments.sum(axis=0)
        class_weights /= class_weights.sum()
        class_weights = 1 / np.sqrt(class_weights + 1e-4)

        weights = assignments @ class_weights 
        weights = weights.reshape(_shape[:-1])
        assignments.reshape(_shape)
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
        return (log_spectrogram > threshold).astype(float)

    @staticmethod
    def source_magnitude_weights(source_spectrograms):
        weights = [
            self.magnitude_weights(source_spectrograms[..., i])
            for i in range(source_spectrograms.shape[-1])
        ]
        weights = np.stack(source_weights, axis=-1)
        weights = source_weights.max(axis=-1)
        return weights

    def _load_audio_file(self, file_path: str) -> AudioSignal:
        """Loads audio file at given path. Uses 
        Args:
            file_path - relative or absolute path to file to load

        Returns:
            tuple of loaded audio w/ shape ? and sample rate
        """
        if os.path.splitext(file_path)[-1] == '.wav':
            rate, audio = wavfile.read(file_path)
            audio = audio.astype(np.float32, order='C') / 32768.0
            audio_signal = AudioSignal(audio_data_array=audio, sample_rate=self.options['sample_rate'])
        else:
            audio_signal = AudioSignal(file_path, sample_rate=self.options['sample_rate'])
        audio_signal.path_to_input_file = file_path
        audio_signal.stft_params.window_length = self.options['n_fft']
        audio_signal.stft_params.hop_length = self.options['hop_length']
        return audio_signal