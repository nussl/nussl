"""
While *nussl* does not come with any data sets, it does have the capability to interface with
many common source separation data sets used within the MIR and speech separation communities.
These data set "hooks" subclass BaseDataset and by default return AudioSignal objects in
labeled dictionaries for ease of use. Transforms can be applied to these datasets for use
in machine learning pipelines.
"""
import os
import warnings

import numpy as np

from ..core import efz_utils, constants, utils
from .. import AudioSignal
import musdb
from .base_dataset import BaseDataset, DataSetException
import jams

class MUSDB18(BaseDataset):
    """
    Hook for MUSDB18. Uses the musdb.DB object to access the
    dataset. If ``download=True``, then the 7s snippets of each track
    are downloaded to ``self.folder``. If no folder is given, then
    the tracks are downloaded to ~/.nussl/musdb18. 
    
    Getting an item from this dataset with no transforms returns the 
    following dictionary:

    .. code-block:: none

        {
            'mix': [AudioSignal object containing mix audio],
            'source': {
                'bass': [AudioSignal object containing vocals],
                'drums': [AudioSignal object containing drums],
                'other': [AudioSignal object containing other],
                'vocals': [AudioSignal object containing vocals],
            }
            'metadata': {
                'labels': ['bass', 'drums', 'other', 'vocals']
            }
        }


    
    Args:
        folder (str, optional): Location that should be processed to produce the 
            list of files. Defaults to None.
        is_wav (bool, optional):  Expect subfolder with wav files for each source 
            instead of stems, defaults to False.
        download (bool, optional): Download sample version of MUSDB18 which 
            includes 7s excerpts. Defaults to False.
        subsets (list, optional): Select a musdb subset train or test. 
            Defaults to ['train', 'test'] (all tracks).
        split (str, optional): When subset train is loaded, split selects the 
            train/validation split. split=’train’ loads the training split, 
            `split=’valid’ loads the validation split. split=None applies no 
            splitting. Defaults to None.
        **kwargs: Any additional arguments that are passed up to BaseDataset 
            (see ``nussl.datasets.BaseDataset``).
    """
    
    def __init__(self, folder=None, is_wav=False, download=False,
            subsets=['train', 'test'], split=None, **kwargs):
        if folder is None:
            folder = os.path.join(
                constants.DEFAULT_DOWNLOAD_DIRECTORY, 'musdb18'
            )
        self.db_args = {
            'is_wav': is_wav,
            'download': download,
            'subsets': subsets,
            'split': split
        }
        super().__init__(folder, **kwargs)

    def get_items(self, folder):
        self.musdb = musdb.DB(root=folder, **self.db_args)
        items = range(len(self.musdb))
        return list(items)

    def process_item(self, item):
        track = self.musdb[item]
        mix, sources = utils.musdb_track_to_audio_signals(track)
        output = {
            'mix': mix,
            'sources': sources,
            'metadata': {
                'labels': ['bass', 'drums', 'other', 'vocals']
            }
        }
        return output


class MixSourceFolder(BaseDataset):
    """
    This dataset expects your data to be formatted in the following way:

    .. code-block:: none

        data/
            mix/
                [file0].wav
                [file1].wav
                [file2].wav
                ...
            [label0]/
                [file0].wav
                [file1].wav
                [file2].wav
                ...
            [label1]/
                [file0].wav
                [file1].wav
                [file2].wav
                ...
            [label2]/
                [file0].wav
                [file1].wav
                [file2].wav
                ...
            ...

    Note that the the filenames match between the mix folder and each source folder.
    The source folder names can be whatever you want. Given a file in the 
    ``self.mix_folder`` folder, this dataset will look up the corresponding files 
    with the same name in the source folders. These are the source audio files. 
    The sum of the sources should equal the mixture. Each source will be labeled 
    according to the folder name it comes from.

    Getting an item from this dataset with no transforms returns the 
    following dictionary:

    .. code-block:: none

        {
            'mix': [AudioSignal object containing mix audio],
            'source': {
                '[label0]': [AudioSignal object containing label0 audio],
                '[label1]': [AudioSignal object containing label1 audio],
                '[label2]': [AudioSignal object containing label2 audio],
                '[label3]': [AudioSignal object containing label3 audio],
                ...
            }
            'metadata': {
                'labels': ['label0', 'label1', 'label2', 'label3']
            }
        }


    Args:
        folder (str, optional): Location that should be processed to produce the 
            list of files. Defaults to None.
        mix_folder (str, optional): Folder to look in for mixtures. Defaults to 'mix'.
        source_folders (list, optional): List of folders to look in for sources. 
            Path is defined relative to folder. If None, all folders other than 
            mix_folder are treated as the source folders. Defaults to None.
        ext (list, optional): Audio extensions to look for in mix_folder. 
            Defaults to ['.wav', '.flac', '.mp3'].
        **kwargs: Any additional arguments that are passed up to BaseDataset 
            (see ``nussl.datasets.BaseDataset``).
    """
    def __init__(self, folder, mix_folder='mix', source_folders=None, sample_rate=None,
            ext=['.wav', '.flac', '.mp3'], **kwargs):
        self.mix_folder = mix_folder
        self.source_folders = source_folders
        self.ext = ext
        super().__init__(folder, **kwargs)

    def get_items(self, folder):
        if self.source_folders is None:
            self.source_folders = sorted([
                f for f in os.listdir(folder)
                if os.path.isdir(os.path.join(folder, f))
                and f != self.mix_folder
            ])

        mix_folder = os.path.join(folder, self.mix_folder)
        items = sorted([
            x for x in os.listdir(mix_folder)
            if os.path.splitext(x)[1] in self.ext
        ])
        return items

    def process_item(self, item):
        mix_path = os.path.join(self.folder, self.mix_folder, item)
        mix = self._load_audio_file(mix_path)
        sources = {}
        for k in self.source_folders:
            source_path = os.path.join(self.folder, k, item)
            if os.path.exists(source_path):
                sources[k] = self._load_audio_file(source_path)
        output = {
            'mix': mix,
            'sources': sources,
            'metadata': {
                'labels': self.source_folders
            }
        }
        return output

class Scaper(BaseDataset):
    """
    Source separation datasets can be generated using Scaper, a library for
    automatic soundscape generation. Datasets that are generated with Scaper
    can be fed into this class easily. Scaper generates a large list of JAMS
    files which specify the parameters of the soundscape. If the soundscape is
    generated with `save_isolated_events=True`, then the audio corresponding
    to each event in the soundscape will be saved as well.

    Below is an example of using Scaper to generate a small dataset of 10 
    mixtures with 2 sources each. The generated dataset can then be immediately
    loaded into an instance of ``nussl.datasets.Scaper`` for integration into
    a training or evaluation pipeline.

    The sources are output in a dictionary that looks like this:

    .. code-block:: none

        data['sources] = {
            '{label}::{count}': AudioSignal,
            '{label}::{count}': AudioSignal,
            ...
        }

    For example:

    .. code-block:: none

        data['sources] = {
            'siren::0': AudioSignal,
            'siren::1': AudioSignal,
            'car_horn::0': AudioSignal,
            ...
        }

    Getting an item from this dataset with no transforms returns the 
    following dictionary:

    .. code-block:: none

        {
            'mix': [AudioSignal object containing mix audio],
            'source': {
                '[label0::count]': [AudioSignal object containing label0 audio],
                '[label1::count]': [AudioSignal object containing label1 audio],
                '[label2::count]': [AudioSignal object containing label2 audio],
                '[label3::count]': [AudioSignal object containing label3 audio],
                ...
            }
            'metadata': {
                'jams': [the content of the jams file used to generate the soundscape]
                'labels': ['label0', 'label1', 'label2', 'label3']
            }
        }


    Example of generating a Scaper dataset and then loading it with nussl:

    >>> n_sources = 2
    >>> n_mixtures = 10
    >>> duration = 3
    >>> ref_db = -40
    >>> fg_path = '/path/to/foreground/'
    >>> output_dir = '/output/path'
    >>> for i in range(n_mixtures):
    >>>     sc = scaper.Scaper(
    >>>         duration, fg_path, fg_path, random_state=i)
    >>>     sc.ref_db = ref_db
    >>>     sc.sr = 16000
    >>>     for j in range(n_sources):
    >>>         sc.add_event(
    >>>             label=('choose', []),
    >>>             source_file=('choose', []),
    >>>             source_time=('const', 0),
    >>>             event_time=('const', 0),
    >>>             event_duration=('const', duration),
    >>>             snr=('const', 0),
    >>>             pitch_shift=None,
    >>>             time_stretch=None
    >>>         )
    >>>     audio_path = os.path.join(output_dir, f'{i}.wav')
    >>>     jams_path = os.path.join(output_dir, f'{i}.jams')
    >>>     sc.generate(audio_path, jams_path, save_isolated_events=True)
    >>> dataset = nussl.datasets.Scaper(output_dir)
    >>> dataset[0] # contains mix, sources, and metadata corresponding to 0.jams.
        
    Raises:
        DataSetException: if Scaper dataset wasn't saved with isolated event audio.
    """
    def get_items(self, folder):
        items = sorted([
            x for x in os.listdir(folder)
            if os.path.splitext(x)[1] in ['.jams']
        ])
        return items

    def process_item(self, item):
        jam = jams.load(os.path.join(self.folder, item))
        ann = jam.annotations.search(namespace='scaper')[0]
        mix_path = ann.sandbox.scaper['soundscape_audio_path']
        source_paths = ann.sandbox.scaper['isolated_events_audio_path']

        if not source_paths:
            raise DataSetException(
                "No paths to isolated events found! Did you generate "
                "the soundscape with save_isolated_events=True?")

        mix = self._load_audio_file(mix_path)
        sources = {}

        for event_spec, event_audio_path in zip(ann, source_paths):
            label = event_spec.value['label']
            label_count = 0
            for k in sources:
                if label in k:
                    label_count += 1
            label = f"{label}::{label_count}"
            sources[label] = self._load_audio_file(event_audio_path)

        output = {
            'mix': mix,
            'sources': sources,
            'metadata': {
                'scaper': jam,
                'labels': ann.sandbox.scaper['fg_labels'],
            }
        }
        return output
