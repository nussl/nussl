import tempfile
from .audio_signal import AudioSignal

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