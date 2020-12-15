"""
While *nussl* does not come with any data sets, it does have the capability to interface with
many common source separation data sets used within the MIR and speech separation communities.
These data set "hooks" subclass BaseDataset and by default return AudioSignal objects in
labeled dictionaries for ease of use. Transforms can be applied to these datasets for use
in machine learning pipelines.
"""
import os
import yaml
from itertools import chain
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from .. import musdb
import numpy as np
import jams
from pretty_midi import PrettyMIDI

from ..core import constants, utils
from .base_dataset import BaseDataset, DataSetException


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
    DATASET_HASHES = {
        "musdb": "56777516ad56fe6a8590badf877e6be013ff932c010e0fbdb0aba03ef878d4cd",
    }
    
    def __init__(self, folder=None, is_wav=False, download=False,
                 subsets=None, split=None, **kwargs):
        subsets = ['train', 'test'] if subsets is None else subsets
        if folder is None:
            folder = os.path.join(
                constants.DEFAULT_DOWNLOAD_DIRECTORY, 'musdb18'
            )
        self.musdb = musdb.DB(root=folder, is_wav=is_wav, download=download, 
                              subsets=subsets, split=split)
        super().__init__(folder, **kwargs)
        self.metadata['subsets'] = subsets
        self.metadata['split'] = split

    def get_items(self, folder):
        items = range(len(self.musdb))
        return list(items)

    def process_item(self, item):
        track = self.musdb[item]
        mix, sources = utils.musdb_track_to_audio_signals(track)
        self._setup_audio_signal(mix)
        for source in list(sources.values()):
            self._setup_audio_signal(source)
        
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
    def __init__(self, folder, mix_folder='mix', source_folders=None,
                 ext=None, make_mix=False, **kwargs):
        self.mix_folder = mix_folder
        self.source_folders = source_folders
        self.ext = ['.wav', '.flac', '.mp3'] if ext is None else ext
        self.make_mix = make_mix
        super().__init__(folder, **kwargs)

    def get_items(self, folder):
        if self.source_folders is None:
            self.source_folders = sorted([
                f for f in os.listdir(folder)
                if os.path.isdir(os.path.join(folder, f))
                and f != self.mix_folder
            ])

        if self.make_mix:
            mix_folder = os.path.join(folder, self.source_folders[0])
        else:
            mix_folder = os.path.join(folder, self.mix_folder)
        items = sorted([
            x for x in os.listdir(mix_folder)
            if os.path.splitext(x)[1] in self.ext
        ])
        return items

    def get_mix_and_sources(self, item):
        sources = {}
        for k in self.source_folders:
            source_path = os.path.join(self.folder, k, item)
            if os.path.exists(source_path):
                sources[k] = self._load_audio_file(source_path)
        
        if self.make_mix:
            mix = sum(list(sources.values()))
        else:
            mix_path = os.path.join(self.folder, self.mix_folder, item)
            mix = self._load_audio_file(mix_path)
        return mix, sources

    def process_item(self, item):
        mix, sources = self.get_mix_and_sources(item)
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

    def _get_info_from_item(self, item):
        jam = jams.load(os.path.join(self.folder, item))
        ann = jam.annotations.search(namespace='scaper')[0]
        mix_path = ann.sandbox.scaper['soundscape_audio_path']
        source_paths = ann.sandbox.scaper['isolated_events_audio_path']
        return jam, ann, mix_path, source_paths

    def process_item(self, item):
        jam, ann, mix_path, source_paths = self._get_info_from_item(item)
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

class OnTheFly(BaseDataset):
    """
    Hook for a dataset that creates mixtures on the fly from source
    data. The function that creates the mixture is a closure which
    is defined by the end-user. The number of mixtures in the 
    dataset is also defined by the end-user. The mix closure function
    should take two arguments - the dataset object and the index of the 
    item being processed - and the output of the mix closure should be a 
    dictionary containing at least a 'mix', 'sources' and (optionally) 
    a 'metadata' key, or other keys that can be defined up to you. 
    Here's an example of a closure, which can be configured via 
    variable scoping:
    
    >>>  def make_sine_wave(freq, sample_rate, duration):
    >>>      dt = 1 / sample_rate
    >>>      x = np.arange(0.0, duration, dt)
    >>>      x = np.sin(2 * np.pi * freq * x)
    >>>      return x
    >>>  n_sources = 2
    >>>  duration = 3
    >>>  sample_rate = 44100
    >>>  min_freq, max_freq = 110, 1000
    >>>  def make_mix(dataset, i):
    >>>      sources = {}
    >>>      freqs = []
    >>>      for i in range(n_sources):
    >>>          freq = np.random.randint(min_freq, max_freq)
    >>>          freqs.append(freq)
    >>>          source_data = make_sine_wave(freq, sample_rate, duration)
    >>>          source_signal = dataset._load_audio_from_array(
    >>>              audio_data=source_data, sample_rate=sample_rate)
    >>>          sources[f'sine{i}'] = source_signal * 1 / n_sources
    >>>      mix = sum(sources.values())
    >>>      output = {
    >>>          'mix': mix,
    >>>          'sources': sources,
    >>>          'metadata': {
    >>>              'frequencies': freqs    
    >>>          }    
    >>>      }
    >>>      return output
    >>>  dataset = nussl.datasets.OnTheFly(make_mix, 10)

    Args:
        mix_closure (function): A closure that determines how to create
          a single mixture, given the index. It has a strict input 
          signature (the index is given as an int) and a strict output
          signature (a dictionary containing a 'mix' and 'sources') key.
        num_mixtures (int): Number of mixtures that will be created on
          the fly. This determines one 'run' thrugh the dataset, or an 
          epoch.
        kwargs: Keyword arguments to BaseDataset.
    """
    def __init__(self, mix_closure, num_mixtures, **kwargs):
        self.num_mixtures = num_mixtures
        self.mix_closure = mix_closure

        super().__init__('none', **kwargs)
        self.metadata['num_mixtures'] = num_mixtures

    def get_items(self, folder):
        return list(range(self.num_mixtures))
    
    def process_item(self, item):
        output = self.mix_closure(self, item)
        if not isinstance(output, dict):
            raise DataSetException("output of mix_closure must be a dict!")
        if 'mix' not in output or 'sources' not in output:
            raise DataSetException(
                "output of mix_closure must be a dict containing "
                "'mix', 'sources' as keys!")
        return output

class FUSS(Scaper):
    """
    The Free Universal Sound Separation (FUSS) Dataset is a database of arbitrary 
    sound mixtures and source-level references, for use in experiments on 
    arbitrary sound separation. 

    This is the official sound separation data for the DCASE2020 Challenge Task 4: 
    Sound Event Detection and Separation in Domestic Environments.

    This is a hook for reading in this dataset, and making sure that the mix and 
    source paths are massaged to be relative paths.

    References:

    [1]  Scott Wisdom, Hakan Erdogan, Daniel P. W. Ellis, Romain Serizel, 
    Nicolas Turpault, Eduardo Fonseca, Justin Salamon, Prem Seetharaman, 
    John R. Hershey, "What's All the FUSS About Free Universal Sound Separation 
    Data?", 2020, in preparation.

    [2] Eduardo Fonseca, Jordi Pons, Xavier Favory, Frederic Font Corbera, 
    Dmitry Bogdanov, Andrés Ferraro, Sergio Oramas, Alastair Porter, and 
    Xavier Serra. "Freesound Datasets: A Platform for the Creation of Open Audio 
    Datasets." International Society for Music Information Retrieval Conference 
    (ISMIR), pp. 486–493. Suzhou, China, 2017.
    
    Args:
        root (str): Folder where the FUSS data is. Either points to ssdata or 
          ssdata_reverb.
        split (str): Either the ``train``, ``validation``, or ``eval`` split. 
        kwargs: Additional keyword arguments to BaseDataset.
    """
    def __init__(self, root, split='train', **kwargs):
        if split not in ['train', 'validation', 'eval']:
            raise DataSetException(
                f"split '{split}' not one of the accepted splits: "
                f"'train', 'validation', 'eval'.")
        
        folder = os.path.join(root, split)
        super().__init__(folder, sample_rate=16000, strict_sample_rate=True, 
                         **kwargs)
        self.metadata['split'] = split

    def _get_info_from_item(self, item):
        path_to_item = os.path.join(self.folder, item)
        item_base_name = os.path.splitext(item)[0]

        jam = jams.load(path_to_item)
        ann = jam.annotations.search(namespace='scaper')[0]
        mix_path = ann.sandbox.scaper['soundscape_audio_path']
        source_paths = ann.sandbox.scaper['isolated_events_audio_path']

        mix_path = os.path.join(
            self.folder, item_base_name + mix_path.split(item_base_name)[-1])
        for i, source_path in enumerate(source_paths):
            source_paths[i] = os.path.join(
                self.folder, item_base_name + source_path.split(item_base_name)[-1])

        return jam, ann, mix_path, source_paths


class WHAM(MixSourceFolder):
    """
    Hook for the WHAM dataset. Essentially subclasses MixSourceFolder but with presets
    that are helpful for WHAM, which as the following directory structure:

    .. code-block:: none

        [wav8k, wav16k]/
          [min, max]/
            [tr, cv, tt]/
                mix_both/
                mix_clean/
                mix_single/
                noise/
                s1/
                s2/
        wham_noise/
          tr/
          cv/
          tt/
          metadata/

    Args:
        root (str): Root of WHAM directory.
        mix_folder (str): Which folder is the mix? Either 'mix_clean', 'mix_both', or
          'mix_single'.
        mode (str): Either 'min' or 'max' mode.
        split (str): Split to use (tr, cv, or tt).
        sample_rate (int): Sample rate of audio, either 8000 or 16000.
    """
    MIX_TO_SOURCE_MAP = {
        'mix_clean': ['s1', 's2'],
        'mix_both': ['s1', 's2', 'noise'],
        'mix_single': ['s1'],
    }

    DATASET_HASHES = {
        "wav8k": "acd49e0dae066e16040c983d71cc5a8adb903abff6e5cbb92b3785a1997b7547", 
        "wav16k": "5691d6a35382f2408a99594f21d820b58371b5ea061841db37d548c0b8d6ec7f"
    }

    def __init__(self, root, mix_folder='mix_clean', mode='min', split='tr', 
                 sample_rate=8000, **kwargs):
        if mix_folder not in self.MIX_TO_SOURCE_MAP.keys():
            raise DataSetException(
                f"{mix_folder} must be in {list(self.MIX_TO_SOURCE_MAP.keys())}")
        if sample_rate not in [8000, 16000]:
            raise DataSetException(
                f"{sample_rate} not available for WHAM (only 8000 and 16000 Hz allowed)")
        if mode not in ['min', 'max']:
            raise DataSetException(
                f"{mode} not available, only 'min' or 'max' allowed.")
        if split not in ['tr', 'cv', 'tt']:
            raise DataSetException(
                f"{split} not available, must be one of 'tr' (train), "
                f"'cv' (validation), and 'tt' (test)")

        wav_folder = 'wav8k' if sample_rate == 8000 else 'wav16k'
        folder = os.path.join(root, wav_folder, mode, split)
        source_folders = self.MIX_TO_SOURCE_MAP[mix_folder]

        super().__init__(folder, mix_folder=mix_folder, source_folders=source_folders,
                         sample_rate=sample_rate, strict_sample_rate=True, **kwargs)

        self.metadata.update({
            'mix_folder': mix_folder,
            'mode': mode,
            'split': split,
            'wav_folder': wav_folder
        })


class Slakh(BaseDataset):
    """
    Hook for the Slakh dataset. Returns a subset of sources from slakh according to a preprovided recipe.
    Slakh is expected to have the following directory structure:

    folder:
        Track00001:
            stems:
                S01.wav
                S02.wav
                ...
            MIDI:
                S01.mid
                S02.mid
                ...
            all_src.mid
            mix.wav
            metadata.yaml
        Track00002:
            ...

    Items returned from this hook will be a dictionary with the keys "mix" and "sources".
    "mix" will contain an AudioSignal which is a mix of all audio signals found in "sources".
    "sources" will contain a dictionary, where each key is the name of a source found in `recipe`,
    a user defined grouping of MIDI synthesized sounds in Slakh.
    If `make_submix=False` then the value of item["sources"]["source"] is a
    list of AudioSignals categorized "source". For example, where `dataset` is a `Slakh` Object:

    >>> item = dataset[0]
    >>> type(item["sources"]["piano"])
    <class 'list'>
    >>> type(item["sources"]["piano"][0])
    <class 'nussl.core.audio_signal.AudioSignal'>

    If `make_submix=True`, then each value will be a submix of stems categorized as a source.

    >>> type(item["sources"]["piano"])
    <class 'nussl.core.audio_signal.AudioSignal'>

    If `midi=True`, then the returned dictionary will contain two more keys, "midi_mix" and
    "midi_sources". The key "midi_mix" will contain a PrettyMIDI object containing the instruments
    of all PrettyMIDI objects in "midi_sources". See http://craffel.github.io/pretty-midi/
    for more information about PrettyMIDI objects, and how they represent MIDI files.
    The key "midi_sources" will be a dictionary structured similarly to "sources" when `make_submix=False`,
    where the PrettyMIDI object found at self.items[i]["midi_sources"][source][j] will be the
    representation of the MIDI file that was synthesized to become the AudioSignal
    object found at self.items[i]["sources"][source][j]. The MIDI files will NOT be submixed when
    `make_submix=True`; only the audio signals are.

    NOTE: If a MIDI object or audio file is specified by the `metadata.yaml` file, but is not
    found in the midi/audio subfolders, then an empty PrettyMIDI/AudioSignal object will take its place.

    It is HIGHLY recommended that the user uses the LibYAML bindings for significantly better
    performance when using Slakh dataset. If it is installed, it will be automatically used.

    Args:
      root (str): Path to the root of the Slakh directory
      split (str): Split of the slakh directory. Can either be 'train', 'val', 'test', or 'all'.
        Uses the `Slakh2100-orig` split. Default='all'
      recipe (dict): Recipe for selecting and grouping sources in Slakh. Each key is a source type,
        and the value is a list of values of `class_key` that correspond with the source.
        For example, if you want a mix that only consists of bass and piano, and `class_key="program_num"`:
        >>> recipe = {
        >>>     "bass": [32, 33, 34, 35, 36, 37, 38, 39],
        >>>     "piano": [0, 1, 2, 3, 4, 5, 6, 7]
        >>> }
        For mappings between midi numbers and instrument types, please see:
        https://github.com/ethman/slakh-utils/blob/master/midi_inst_values/general_midi_inst_0based.txt
        Default: None, which populates with a default recipe, containing the sources,
        drums, guitar, piano, and bass. A recipe must be defined if `class_key="plugin_name"`
      class_key (str): Metadata key to separate stems by. Can either be "inst_class", "program_num",
        or "plugin_name". Default: "inst_class"
      midi (bool): If True, return PrettyMIDI objects. default=False
      min_acceptable_sources (int): Number of sources a song must have in the recipe to be included in
        `self.get_items()`. default=2
      make_submix (bool): If `True`, make submixes of each source. default=False.
    """
    def __init__(self, root, recipe=None, split='train', class_key="inst_class",
                 min_acceptable_sources=2, midi=False, make_submix=False, transform=None,
                 sample_rate=None, stft_params=None, num_channels=None, strict_sample_rate=True,
                 cache_populated=False):

        self.class_key = class_key
        recipe = Slakh.default_recipe(class_key) if recipe is None else recipe
        self.sources = list(recipe.keys())
        self.recipe = {}
        for key, l in recipe.items():
            for val in l:
                if val in self.recipe.keys():
                    raise ValueError(f"Identifier {val} found in multiple source types!")
                self.recipe[val] = key
        self.midi = midi
        self.make_submix = make_submix

        if min_acceptable_sources <= 0:
            raise ValueError("`min_acceptable_sources` should be positive!")

        self.split = Slakh.get_split(split)

        self.min_acceptable_sources = min_acceptable_sources

        super().__init__(root, transform, sample_rate, stft_params, num_channels,
            strict_sample_rate, cache_populated)

        if not self.items:
            raise DataSetException(
                f"No Slakh tracks were found with the recipe: {recipe} "
                + f"and minimum acceptable instruments of {min_acceptable_sources}."
            )

    def get_items(self, folder):
        # Remove tracks with less than `self.min_acceptable_sources`
        def acceptable_source(path):
            id_ = int(path[5:])
            if id_ not in self.split:
                return False
            path = os.path.join(folder, path)
            with open(os.path.join(path, 'metadata.yaml'), 'r') as file:
                metadata = yaml.load(file, Loader=Loader)
            sources = set()
            for stem, data in metadata["stems"].items():
                if data[self.class_key] in self.recipe.keys():
                    sources.add(self.recipe[data[self.class_key]])
            return len(sources) >= self.min_acceptable_sources

        trackpaths = [os.path.join(folder, f) for f in os.listdir(folder)
                      if acceptable_source(f)]

        return trackpaths


    def submix(self, sources, num_frames):
        for source, values in sources.items():
            sources[source] = (
                sum(values) if values else
                self._load_audio_from_array(np.zeros((1, num_frames),
                                                      dtype=np.float32)))


    def _find_source(self, stem_dict):
        midi_num = stem_dict[self.class_key]
        key = self.recipe.get(midi_num, None)
        return key

    def _get_empty(self, num_frames):
        return self._load_audio_from_array(np.zeros((1, num_frames), np.float32))

    def _get_mix_and_souces_audio(self, audio_dir, stems_dict):
        stempaths = {os.path.splitext(file)[0]:
                     os.path.join(audio_dir, file)
                     for file in os.listdir(audio_dir)}
        num_frames = self._load_audio_file(
            list(stempaths.values())[0]
        ).signal_length

        sources = {}
        for source in self.sources:
            sources[source] = []

        stems = sorted(list(stems_dict.keys()))
        for stem in stems:
            source_type = self._find_source(stems_dict[stem])
            if source_type is None:
                continue
            stempath = stempaths.get(stem, None)
            if stempath is None:
                sources[source_type].append(self._get_empty(num_frames))
            else:
                sources[source_type].append(
                    self._load_audio_file(stempath)
                )

        if self.make_submix:
            self.submix(sources, num_frames)
            mix = sum([signal for signal in sources.values()])
        else:
            mix = sum([sum(signals) for signals in sources.values() if signals])
        return mix, sources

    def _get_mix_and_sources_midi(self, midi_dir, stems_dict, midi_mix_path):
        sources = {}
        midipaths = {os.path.splitext(file)[0]:
                     os.path.join(midi_dir, file)
                     for file in os.listdir(midi_dir)}
        for source in self.sources:
            sources[source] = []
        midi_mix = PrettyMIDI(midi_mix_path)
        midi_mix.instruments = []
        stems = sorted(list(stems_dict.keys()))
        for stem in stems:
            source_type = self._find_source(stems_dict[stem])
            if source_type is None:
                continue
            midipath = midipaths.get(stem, None)
            src_midi = PrettyMIDI(midipath)
            sources[source_type].append(src_midi)
            midi_mix.instruments.extend(src_midi.instruments)
        return midi_mix, sources

    def process_item(self, srcs_dir):
        # Use the file's metadata and the submix recipe to gather all the
        # sources together.
        src_metadata = yaml.load(open(os.path.join(srcs_dir, 'metadata.yaml'), 'r'), Loader=Loader)
        _, audio_dir = os.path.split(src_metadata["audio_dir"])
        _, midi_dir = os.path.split(src_metadata["midi_dir"])

        item = {}
        item['mix'], item['sources'] = self._get_mix_and_souces_audio(
            os.path.join(srcs_dir, audio_dir),
            src_metadata['stems']
        )
        if self.midi:
            item['midi_mix'], item['midi_sources'] = self._get_mix_and_sources_midi(
                os.path.join(srcs_dir, midi_dir),
                src_metadata['stems'],
                os.path.join(srcs_dir, "all_src.mid")
            )

        return item

    @staticmethod
    def get_split(split_name):
        splits = {
            'train': range(1, 1501),
            'val': range(1501, 1876),
            'test': range(1876, 2101),
            'all': range(1, 2101)
        }
        split = splits.get(split_name, None)
        if split is None:
            raise ValueError(f"`split`: {split_name} not a valid split."
                f"`split` must be in {list(splits.keys())}")

        return split

    @staticmethod
    def default_recipe(class_key):
        if class_key == "program_num":
            recipe = {
                'piano': range(8),
                'guitar': range(24, 32),
                'bass': range(32, 40),
                'drums': [128]
            }
        elif class_key == "inst_class":
            recipe = {
                'piano': ["Piano"],
                'guitar': ["Guitar"],
                'bass': ["Bass"],
                'drums': ["Drums"]
            }
        elif class_key == "plugin_name":
            raise ValueError("There is no default recipe for `class_key='plugin_name'`."
                "A recipe must be defined.")
        else:
            raise ValueError(f"Invalid class_key: {class_key}")
        return recipe
