from .base_dataset import BaseDataset
import librosa
import jams
import os
import scaper
import numpy as np

class Scaper(BaseDataset):
    def __init__(self, folder, options=None):
        super(Scaper, self).__init__(folder, options)

        #initialization
        jam_file = self.files[0]
        jam = jams.load(jam_file)
        
        if len(self.options['source_labels']) == 0:
            all_classes = jam.annotations[0]['sandbox']['scaper']['fg_labels']
            classes = jam.annotations[0]['sandbox']['scaper']['fg_spec'][0][0][1]
            if len(classes) <= 1:
                classes = all_classes
            self.options['source_labels'] = classes
        
        for i in range(len(self.options['group_sources'])):
            self.options['source_labels'].append(f'group{i}')

    def get_files(self, folder):
        files = sorted([os.path.join(folder, x) for x in os.listdir(folder) if '.json' in x])
        return files        

    def load_audio_files(self, file_name):
        mix, sr = self._load_audio_file(file_name[:-4] + 'wav')
        jam = jams.load(file_name)
        data = jam.annotations[0]['data']['value']           
        classes = self.options['source_labels']
        source_dict = {}

        for d in data:
            if d['role'] == 'foreground':
                source_path = d['saved_source_file']
                source_path = os.path.join(self.folder, source_path.split('/')[-1])
                source_dict[d['label']] = self._load_audio_file(source_path)[0]

        for i, group in enumerate(self.options['group_sources']):
            combined = []
            for label in group:
                combined.append(source_dict[label])
                source_dict.pop(label)
            source_dict[f'group{i}'] = sum(combined)

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