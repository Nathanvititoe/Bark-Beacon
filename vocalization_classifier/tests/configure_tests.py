import pytest
import sys
import pathlib
import os
import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# defined sampling rate for tests
@pytest.fixture(scope="session")
def sr():
    return 16_000

# defined sample duration for tests
@pytest.fixture(scope="session")
def duration():
    return 1

@pytest.fixture(autouse=True)
def patch_config(monkeypatch, sr):
    import src.prep_data.preprocess as pp
    import config

    # ensure sample rate config applies correctly
    monkeypatch.setattr(pp, "SAMPLE_RATE", sr, raising=False)

    # ensure duration config applies correctly
    monkeypatch.setattr(pp, "DURATION_SEC", duration, raising=False)

    # ensure fourier transform config applies correctly
    monkeypatch.setattr(pp, "N_FFT", 512, raising=False)

    # ensure length config applies correctly
    monkeypatch.setattr(pp, "HOP_LENGTH", 160, raising=False)

    # ensure mel num config applies correctly
    monkeypatch.setattr(pp, "N_MELS", 32, raising=False)

    # ensure duration width config applies correctly
    monkeypatch.setattr(pp, "TARGET_W", 64, raising=False)

    # ensure visualization is off so that it doesnt try to display images
    monkeypatch.setattr(pp, "SHOW_VISUALS", False, raising=False)

    # get test files
    audio_dir = getattr(config, "TEST_AUDIO_DIR", None)
    if audio_dir:
        audio_dir = str(audio_dir).strip()
        if audio_dir and os.path.isdir(audio_dir):
            # replace dataset dir with test audio files (in testing)
            monkeypatch.setattr(pp, "AUDIO_ROOT_PATH", audio_dir, raising=False)
    yield


# gives pytest the ability to write wav files to the mock audio root dir
@pytest.fixture
def write_wav(tmp_path):
    def _write(rel_path: str, data: np.ndarray, sr: int):
        out = tmp_path / "audio_root" / rel_path
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, data.astype(np.float32), sr)
        return str(out)

    return _write
