"""
These are optional utilities included in nussl that allow one to embed an AudioSignal
as a playable object in a Jupyter notebook, or to play audio from
the terminal.
"""
from copy import deepcopy
import subprocess
from tempfile import NamedTemporaryFile

import random, string
import importlib_resources as pkg_resources

from . import templates
from .utils import _close_temp_files

multitrack_template = pkg_resources.read_text(templates, 'multitrack.html')


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


def embed_audio(audio_signal, ext='.mp3', display=True):
    """
    Write a numpy array to a temporary mp3 file using ffmpy, then embeds the mp3
    into the notebook.

    Args:
        audio_signal (AudioSignal): AudioSignal object containing the data.
        ext (str): What extension to use when embedding. '.mp3' is more lightweight 
          leading to smaller notebook sizes. Defaults to '.mp3'.
        display (bool): Whether or not to display the object immediately, or to return
          the html object for display later by the end user. Defaults to True.

    Example:
        >>> import nussl
        >>> audio_file = nussl.efz_utils.download_audio_file('schoolboy_fascination_excerpt.wav')
        >>> audio_signal = nussl.AudioSignal(audio_file)
        >>> audio_signal.embed_audio()

    This will show a little audio player where you can play the audio inline in 
    the notebook.      
    """
    audio_signal = deepcopy(audio_signal)
    ffmpy, IPython = _check_imports()
    sr = audio_signal.sample_rate
    tmpfiles = []

    with _close_temp_files(tmpfiles):
        tmp_wav = NamedTemporaryFile(
            mode='w+', suffix='.wav', delete=False)
        tmpfiles.append(tmp_wav)
        audio_signal.write_audio_to_file(tmp_wav.name)
        if ext != '.wav' and ffmpy:
            tmp_converted = NamedTemporaryFile(
                mode='w+', suffix=ext, delete=False)
            tmpfiles.append(tmp_wav)
            ff = ffmpy.FFmpeg(
                inputs={tmp_wav.name: None},
                outputs={tmp_converted.name: '-write_xing 0 -codec:a libmp3lame -b:a 128k -y'})
            ff.run()
        else:
            tmp_converted = tmp_wav

        audio_element = IPython.display.Audio(data=tmp_converted.name, rate=sr)
        if display:
            IPython.display.display(audio_element)
    return audio_element


def multitrack(audio_signals, names=None, ext='.mp3', display=True):
    """
    Takes a bunch of audio sources, converts them to mp3 to make them smaller, and
    creates a multitrack audio player in the notebook that lets you
    toggle between the sources and the mixture. Heavily adapted
    from https://github.com/binarymind/multitrackHTMLPlayer,
    designed by Bastien Liutkus.

    Args:
        audio_signals (list): List of AudioSignal objects that add up to the mixture.
        names (list): List of names to give to each object (e.g. foreground, background).
        ext (str): What extension to use when embedding. '.mp3' is more lightweight
          leading to smaller notebook sizes.
        display (bool): Whether or not to display the object immediately, or to return
          the html object for display later by the end user.
    """
    ffmpy, IPython = _check_imports()
    div_id = ''.join(random.choice(string.ascii_uppercase) for _ in range(20))
    _names = None

    if isinstance(audio_signals, dict):
        _names = list(audio_signals.keys())
        audio_signals = [audio_signals[k] for k in _names]

    if names is not None:
        if len(names) != len(audio_signals):
            raise ValueError("len(names) must be equal to len(audio_signals)!")
    else:
        if _names is not None:
            names = _names
        else:
            names = [
                f"{i}:{s.path_to_input_file}"
                for i, s in enumerate(audio_signals)
            ]

    template = (
        f"<div id={div_id} class=audio-container "
        f"preload=auto name={div_id}>")

    for name, signal in zip(names, audio_signals):
        encoded_audio = embed_audio(signal, ext=ext, display=False).src_attr()
        audio_element = (
            f"<audio name='{name}' url={encoded_audio}></audio>")
        template += audio_element

    template += "</div>"
    template += multitrack_template
    template = template.replace('NAME', div_id)

    html = IPython.display.HTML(template)
    if display:
        IPython.display.display(html)
    else:
        return html


def play(audio_signal):
    """
    Plays an audio signal if ffplay from the ffmpeg suite of tools is installed.
    Otherwise, will fail. The audio signal is written to a temporary file
    and then played with ffplay.
    
    Args:
        audio_signal (AudioSignal): AudioSignal object to be played.
    """
    tmpfiles = []
    with _close_temp_files(tmpfiles):
        tmp_wav = NamedTemporaryFile(suffix='.wav', delete=False)
        tmpfiles.append(tmp_wav)
        audio_signal.write_audio_to_file(tmp_wav.name)
        subprocess.call(["ffplay", "-nodisp", "-autoexit", tmp_wav.name])
