from .base_dataset import BaseDataset
import os
import numpy as np
import random

class MixSourceFolder(BaseDataset):
    def __init__(self, folder, options):
        """This dataset expects your data to be formatted in the following way:

            data/
                mix/
                    [file0].wav
                    [file1].wav
                    [file2].wav
                    ...
                s0/
                    [file0].wav
                    [file1].wav
                    [file2].wav
                    ...
                s1/
                    [file0].wav
                    [file1].wav
                    [file2].wav
                    ...
                s2/
                    [file0].wav
                    [file1].wav
                    [file2].wav
                    ...
                ...
        
        Note that the the filenames match between the mix folder and each source folder.
        The only protected folder name here is 'mix'. The source folder names can be
        whatever you want. Given a file in the 'mix' folder, this dataset will look up
        the corresponding files with the same name in the source folders. These are the
        source audio files. The sum of the sources should equal the mixture. Each source
        will be labeled according to the folder name it comes from.
        
        If no mix folder exists, then this class will use the files in the first source
        folder and look for matching files in the other source folder. Then the mix
        will be created by summing the sources on the fly. This can also happen
        if there is a mix folder by setting self.options['sum_sources'] = True.

        Arguments:
            folder: location where all the data lives that is in MixSourceFolder format.
            options: the options for the dataset (see deep/config/defaults/dataset.json)
                for more details.            
        """
        super(MixSourceFolder, self).__init__(folder, options)
        if self.create_mix:
            wav_file = os.path.join(self.folder, self.source_folders[0], self.files[0])
        else:
            wav_file = os.path.join(self.folder, 'mix', self.files[0])
        
        mix = self._load_audio_file(wav_file)
        self.channels_in_mix = mix.num_channels
        self.sum_sources = self.options.pop('sum_sources', False)

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
        channel_indices = list(range(1, self.channels_in_mix))
        random.shuffle(channel_indices)
        channel_indices = [0] + channel_indices[:self.options['num_channels'] - 1]
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
        if self.sum_sources:
            mix_signal = sum(sources)
        return mix_signal, sources, one_hots