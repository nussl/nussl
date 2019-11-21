from .base_dataset import BaseDataset
import librosa
import jams
import os
import numpy as np

class Scaper(BaseDataset):
    def __init__(self, folder, options):
        """This dataset expects that your data was mixed using Scaper, which generates
        mixtures from a given source library. Scaper generates a bunch of JSON files
        which are then parsed by this dataset class. Each JSON file contains all of the
        metadata to find the sources that went into the mixture, the mixture itself (must
        have the same name as the JSON file but with a different extension), and the 
        classes of the sources. This JSON file is processed here to load the mixture
        and sources, and label the sources.

        For now, Scaper datasets must be mixed by this fork of Scaper, kept here:
            https://github.com/pseeth/scaper

        This fork saves the sources alongside the mixture. Datasets created using this
        fork can be immediately used for training a deep source separation model by
        using this Dataset class.
        """
        super(Scaper, self).__init__(folder, options)

        #initialization
        jam_file = self.files[0]
        jam = jams.load(jam_file)
        
        if not self.options['source_labels']:
            self.options['source_labels'] = jam.annotations[0]['sandbox']['scaper']['fg_labels']
        
        for group in self.options['group_sources']:
            group_name = '+'.join(group)
            self.options['source_labels'].append(group_name)

    def get_files(self, folder):
        files = sorted(
            [os.path.join(folder, x.replace('.wav', '.jams')) for x in os.listdir(folder) 
            if '.wav' in x
        ])
        return files

    def load_audio_files(self, file_name):
        mix = self._load_audio_file(file_name.replace('.jams', '.wav'))
        source_folder = file_name.replace('.jams', '_sources')
        jam = jams.load(file_name)
        data = jam.annotations[0]['data']      
        classes = self.options['source_labels']
        source_dict = {}
        lengths = [mix.signal_length]

        for datum in data:
            d = datum.value
            if d['role'] == 'foreground':
                source_path = os.path.join(source_folder, d['audio_path'] + '.wav')
                source_dict[d['label']] = self._load_audio_file(source_path)
                lengths.append(source_dict[d['label']].signal_length)

        min_length = min(lengths)
        mix.audio_data = mix.audio_data[:, :min_length]
        for key in source_dict:
            source_dict[key].audio_data = source_dict[key].audio_data[:, :min_length]

        for i, group in enumerate(self.options['group_sources']):
            combined = []
            for label in group:
                combined.append(source_dict[label])
                source_dict.pop(label)
            group_name = '+'.join(group)
            source_dict[group_name] = sum(combined)

        sources = []
        one_hots = []

        for i, label in enumerate(classes):
            if label in source_dict:
                sources.append(source_dict[label])
                one_hot = np.zeros(len(classes))
                one_hot[classes.index(label)] = 1
                one_hots.append(one_hot)
        one_hots = np.stack(one_hots)
        return mix, sources, one_hots