from .base_dataset import BaseDataset
import os
import numpy as np

class WSJ(BaseDataset):
    def __init__(self, folder, options=None):
        super(WSJ, self).__init__(folder, options)

        wav_file = os.path.join(self.folder, 'mix', self.files[0])
        mix = self._load_audio_file(wav_file)[0]
        self.channels_in_mix = mix.shape[0] if mix.shape[0] < 8 else int(mix.shape[0] / 2)

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

        for speaker in self.speaker_folders:
            speaker_path = os.path.join(self.folder, speaker, wav_file)
            mix_path = os.path.join(self.folder, 'mix', wav_file)

            mix, _ = self._load_audio_file(mix_path)
            source, _ = self._load_audio_file(speaker_path)

            mix = mix[channel_indices]
            source = source[channel_indices]
            sources.append(source)

        return mix, sources, np.eye(self.num_speakers)
