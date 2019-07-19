##################################################
#              Jupyter integration
##################################################

from tempfile import NamedTemporaryFile
import librosa
from .. import ImportErrorClass
from copy import deepcopy

def _check_imports():
    try:
        import ffmpy
    except:
        ffmpy = False

    try:
        import IPython
    except:
        raise ImportError('IPython must be installed in order to use this function!')
    return ffmpy, IPython

def embed_audio(audio_signal, ext = '.mp3'):
    """
	Write a numpy array to a temporary mp3 file using ffmpy, then embeds the mp3 into the notebook.

	Parameters:
	   audio_signal: AudioSignal object containing the data
       ext: What extension to use when embedding. mp3 is more lightweight leading to smaller notebook sizes.

    Example:
        >>> import nussl
        >>> audio_file = nussl.efz_utils.download_audio_file('schoolboy_fascination_excerpt.wav')
        >>> audio_signal = nussl.AudioSignal(audio_file)
        >>> nussl.jupyter_utils.embed_audio(audio_signal)

        This will show a little audio player where you can play the audio inline in the notebook.
	"""
    audio_signal = deepcopy(audio_signal)
    ffmpy, IPython = _check_imports()
    sr = audio_signal.sample_rate
    d = audio_signal.audio_data.T

    tmp_wav = NamedTemporaryFile(mode='w+', suffix='.wav')
    audio_signal.write_audio_to_file(tmp_wav.name)
    if ext != '.wav' and ffmpy:
        tmp_converted = NamedTemporaryFile(mode='w+', suffix=ext)
        ff = ffmpy.FFmpeg(
            inputs={tmp_wav.name: None},
            outputs={tmp_converted.name: '-write_xing 0 -codec:a libmp3lame -b:a 128k -y'})
        ff.run()
    else:
        tmp_converted = tmp_wav
    IPython.display.display(IPython.display.Audio(data=tmp_converted.name, rate = sr))
    if ext != '.wav':
        tmp_converted.close()
    tmp_wav.close()