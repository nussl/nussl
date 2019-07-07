from .base_dataset import BaseDataset
import os
import numpy as np

class MixSourceFolder(BaseDataset):
    def __init__(self, folder, options=None):
        super(MixSourceFolder, self).__init__(folder, options)
        if self.create_mix:
            wav_file = os.path.join(self.folder, self.source_folders[0], self.files[0])
        else:
            wav_file = os.path.join(self.folder, 'mix', self.files[0])
        
        mix = self._load_audio_file(wav_file)
        self.channels_in_mix = mix.num_channels

    def get_files(self, folder):
        self.source_folders = sorted([
            x for x in os.listdir(folder) 
            if os.path.isdir(os.path.join(folder, x)) 
            and 'mix' not in x and 'cache' not in x
        ])
        if not self.options['source_labels']:
            self.options['source_labels'] = self.source_folders

        for group in self.options['group_sources']:
            group_name = '+'.join(group)
            self.options['source_labels'].append(group_name)

        self.num_sources = len(self.source_folders)
        if os.path.exists(os.path.join(folder, 'mix')):
            self.create_mix = False
            files = sorted([x for x in os.listdir(os.path.join(folder, 'mix')) if '.wav' in x])
        else:
            self.create_mix = True
            files = sorted([x for x in os.listdir(os.path.join(folder, self.source_folders[0])) if '.wav' in x])
        return files

    def load_audio_files(self, wav_file):
        channel_indices = np.arange(self.channels_in_mix)
        np.random.shuffle(channel_indices)
        channel_indices = channel_indices[:self.options['num_channels']]
        source_dict = {}
        classes = self.options['source_labels']

        for source in self.source_folders:
            source_path = os.path.join(self.folder, source, wav_file)
            if os.path.isfile(source_path):
                source_signal = self._load_audio_file(source_path)
                source_signal.audio_data = source_signal.audio_data[channel_indices]
                source_dict[source] = source_signal

        if self.create_mix:
            mix_signal = sum([source_dict[x] for x in source_dict])
        else:
            mix_path = os.path.join(self.folder, 'mix', wav_file)
            mix_signal = self._load_audio_file(mix_path)
            mix_signal.audio_data = mix_signal.audio_data[channel_indices]

        for i, group in enumerate(self.options['group_sources']):
            combined = []
            for label in group:
                combined.append(source_dict[label])
                source_dict.pop(label)
            group_name = '+'.join(group)
            source_dict[group_name] = sum(combined)

        sources = []
        one_hots = []

        for i, label in enumerate(self.options['source_labels']):
            if label in source_dict:
                sources.append(source_dict[label])
                one_hot = np.zeros(len(classes))
                one_hot[classes.index(label)] = 1
                one_hots.append(one_hot)
        
        one_hots = np.stack(one_hots)
        return mix_signal, sources, one_hots