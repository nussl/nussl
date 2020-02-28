from torch.utils.data import Dataset
from .. import AudioSignal, STFTParams

# pre-cache transforms (taking stft, etc.)
# post-cache transforms (getting offsets, etc.)

# caching = first class citizen of dataset

#'log_spectrogram': log_spectrogram,
# 'magnitude_spectrogram': mix_magnitude,
# 'assignments': assignments,
# 'source_spectrograms': source_magnitudes,

class BaseDataset(Dataset):
    def __init__(self, folder, sample_rate=None, transforms=None):
        self.folder = folder
        self.transforms = transforms
        self.items = self.get_items(self.folder)
        if not isinstance(self.items, list):
            raise DataSetException("Output of self.get_items must be a list!")

        self.sample_rate = sample_rate

    def get_items(self, folder):
        """This function must be implemented by whatever class inherits BaseDataset.
        It should return a list of items in the given folder, each of which is 
        processed by load_audio_files in some way to produce mixes, sources, class
        labels, etc.

        Args:
            folder - location that should be processed to produce the list of files.

        Returns:
            list: list of items that should be processed
        """
        raise NotImplementedError()

    def __len__(self):
        """
        Gets the length of the dataset (the number of items that will be processed).

        Returns:
            int: Length of the dataset (``len(self.items)``).
        """
        return len(self.items)

    def __getitem__(self, i):
        """
        Processes a single item in ``self.items`` using ``self.process_item``.
        The output of ``self.process_item`` is further passed through bunch of
        of transforms if they are defined in parallel. If you want to have
        a set of transforms that depend on each other, then you should compose them
        into a single transforms and then pass it into here. The output of each
        transform is added to an output dictionary which is returned by this
        function.
        
        Args:
            i (int): Index of the dataset to return. Indexes ``self.items``.

        Returns:
            dict: Dictionary with keys and values corresponding to the processed
                item after being put through the set of transforms (if any are
                defined).
        """
        output = {}
        processed_item = self.process_item(self.items[i])

        if not isinstance(processed_item, dict):
            raise DataSetException(
                "The output of process_item must be a dictionary!")

        output.update(processed_item)

        if self.transforms:
            for transform in self.transforms:
                transform_output = transform(processed_item)
                if not isinstance(transform_output, dict):
                    raise DataSetException(
                        "The output of every transform must be a dictionary!")
                output.update(transform_output)

        return output

    def process_item(self, item):
        """Each file returned by get_items is processed by this function. For example,
        if each file is a json file containing the paths to the mixture and sources, 
        then this function should parse the json file and load the mixture and sources
        and return them.

        Exact behavior of this functionality is determined by implementation by subclass.

        Args:
            item (object): the item that will be processed by this function. Input depends
                on implementation of ``self.get_items``.

        Returns:
            This should return a dictionary that gets processed by the transforms.
        """
        raise NotImplementedError()

    def _load_audio_file(self, path_to_audio_file):
        """
        Loads audio file at given path. Uses AudioSignal to load the audio data
        from disk.

        Args:
            path_to_audio_file - relative or absolute path to file to load

        Returns:
            AudioSignal: loaded AudioSignal object of path_to_audio_file
        """
        audio_signal = AudioSignal(
            path_to_audio_file, sample_rate=self.sample_rate)
        return audio_signal
    
    def _load_audio_from_array(self, audio_data):
        """
        Loads the audio data into an AudioSignal object with the appropriate 
        sample rate.
        
        Args:
            audio_data (np.ndarray): numpy array containing the samples containing
                the audio data.
        
        Returns:
            AudioSignal: loaded AudioSignal object of audio_data
        """

        audio_signal = AudioSignal(
            audio_data_array=audio_data, sample_rate=self.sample_rate
        )
        return audio_signal

class DataSetException(Exception):
    """
    Exception class for errors when working with data sets in nussl.
    """
    pass
