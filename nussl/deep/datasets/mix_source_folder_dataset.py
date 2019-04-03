from .base_dataset import BaseDataset
import os
import numpy as np

class MixSourceFolder(BaseDataset):
    def __init__(self, folder, options=None):
        super(MixSourceFolder, self).__init__(folder, options)

        wav_file = os.path.join(self.folder, 'mix', self.files[0])
        mix = self._load_audio_file(wav_file)
        self.channels_in_mix = mix.num_channels

    def get_files(self, folder):
        files = sorted([x for x in os.listdir(os.path.join(folder, 'mix')) if '.wav' in x])

        self.source_folders = sorted([
            x for x in os.listdir(folder) 
            if os.path.isdir(os.path.join(folder, x)) 
            and 'mix' not in x and 'cache' not in x
        ])
        self.num_sources = len(self.source_folders)
        return files

    def load_audio_files(self, wav_file):
        sources = []
        channel_indices = np.arange(self.channels_in_mix)
        np.random.shuffle(channel_indices)
        channel_indices = channel_indices[:self.options['num_channels']]

        mix_path = os.path.join(self.folder, 'mix', wav_file)
        mix_signal = self._load_audio_file(mix_path)
        mix_signal.audio_data = mix_signal.audio_data[channel_indices]

        for source in self.source_folders:
            source_path = os.path.join(self.folder, source, wav_file)
            if os.path.isfile(source_path):
                source_signal = self._load_audio_file(source_path)
                source_signal.audio_data = source_signal.audio_data[channel_indices]
                sources.append(source_signal)

        return mix_signal, sources, np.eye(self.num_sources)
