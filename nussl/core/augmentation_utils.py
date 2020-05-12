import tempfile
from .audio_signal import AudioSignal
import ffmpeg
from .utils import _close_temp_files

def apply_ffmpeg_filter(audio_signal, _filter, silent=True, **kwargs):
    tmpfiles = []
    with _close_temp_files(tmpfiles):
        input_tempfile = tempfile.NamedTemporaryFile(suffix=".wav")
        output_tempfile = tempfile.NamedTemporaryFile(suffix=".wav")
        tmpfiles.append(input_tempfile)
        tmpfiles.append(output_tempfile)

        audio_signal.write_audio_to_file(input_tempfile)

        if silent:
            input_kwargs = {'loglevel': 'quiet'}
        else:
            input_kwargs = {}
        
        output = (ffmpeg
            .input(input_tempfile.name, **input_kwargs)
            .filter(_filter, **kwargs)
            .output(output_tempfile.name)
            .overwrite_output()
            .run()
        )
        
        augmented_signal = AudioSignal(path_to_input_file=output_tempfile.name)
        output_tempfile.close()
        input_tempfile.close()
    return augmented_signal
