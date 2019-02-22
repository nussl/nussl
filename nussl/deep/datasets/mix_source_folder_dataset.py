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
        files = [x for x in os.listdir(os.path.join(folder, 'mix')) if '.wav' in x]
        files = sorted([x for x in os.listdir(os.path.join(folder, 'mix')) if '.wav' in x])

        self.speaker_folders = sorted([x for x in os.listdir(folder) if 's' in x and x != 'scaling.mat'])
        self.num_speakers = len(self.speaker_folders)
        return files

    def load_audio_files(self, wav_file):
        sources = []
        channel_indices = np.arange(self.channels_in_mix)
        np.random.shuffle(channel_indices)
        channel_indices = channel_indices[:self.options['num_channels']]

        mix_path = os.path.join(self.folder, 'mix', wav_file)
        mix_signal = self._load_audio_file(mix_path)
        mix_signal.audio_data = mix_signal.audio_data[channel_indices]

        for speaker in self.speaker_folders:
            speaker_path = os.path.join(self.folder, speaker, wav_file)

            source_signal = self._load_audio_file(speaker_path)
            source_signal.audio_data = source_signal.audio_data[channel_indices]
            sources.append(source_signal)

        return mix_signal, sources, np.eye(self.num_speakers)
