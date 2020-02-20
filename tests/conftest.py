import pytest
from nussl import efz_utils
import tempfile
import os

fix_dir = os.path.expanduser('~/.nussl/tests/')

@pytest.fixture(scope="module")
def benchmark_audio():
    audio_files = {}
    keys = ['K0140.wav', 'K0149.wav', 'dev1_female3_inst_mix.wav']
    with tempfile.TemporaryDirectory() as tmp_dir:
        _dir = tmp_dir if fix_dir is None else fix_dir
        for k in keys:
            audio_files[k] = efz_utils.download_audio_file(k, _dir)
        yield audio_files