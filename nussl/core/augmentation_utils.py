import tempfile
from .audio_signal import AudioSignal
import ffmpeg
from .utils import _close_temp_files

def save_audio_signal_to_tempfile(audio_signal, extension=".wav"):
    """
    Saves AudioSignal object as a audio tempfile, default in ".wav" format.
    Args:
        audio_signal: AudioSignal object to be saved
        extension: extension to save audiofile as.
    Returns:
            audio_temp = tempfile.NamedTemporaryFile(suffix=extension)
: tempfile object
        audio_signal: name of the tempfile object
    """

    audio_temp = tempfile.NamedTemporaryFile(suffix=extension)
    audio_temp_name = audio_temp.name
    audio_signal.write_audio_to_file(audio_temp)

    return audio_temp, audio_temp_name

def make_empty_audio_file(extension=".wav"):
    """
    Returns an empty tempfile object with an extenion
    """
    temp = tempfile.NamedTemporaryFile(suffix=extension)
    temp_name = temp.name
    return temp, temp_name

def read_audio_tempfile(temp):
    """
    Takes a tempfile, reads the data into an AudioSignal object, 
    closes the tempfile, and returns the AudioSignal.
    Args:
        temp: A tempfile object with audio data
    Returns:
        audio_signal: An audio signal object with the audio data 
        from temp
    """
    audio_signal = AudioSignal(path_to_input_file=temp.name)
    temp.close()
    return audio_signal

def apply_ffmpeg_filter(audio_signal, _filter, silent=True, **kwargs):
    tmpfiles = []
    with _close_temp_files(tmpfiles):
        audio_tempfile, audio_tempfile_name = save_audio_signal_to_tempfile(audio_signal)
        output_tempfile, output_tempfile_name = make_empty_audio_file()
        tmpfiles.append(audio_tempfile)
        tmpfiles.append(output_tempfile)

        if silent:
            input_kwargs = {'loglevel': 'quiet'}
        else:
            input_kwargs = {}
        
        output = (ffmpeg
            .input(audio_tempfile_name, **input_kwargs)
            .filter(_filter, **kwargs)
            .output(output_tempfile_name)
            .overwrite_output()
            .run()
        )
        
        augmented_signal = read_audio_tempfile(output_tempfile)
    return augmented_signal
