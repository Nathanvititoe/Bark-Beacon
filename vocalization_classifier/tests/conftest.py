import pytest
import sys
import pathlib
import os
import numpy as np
import soundfile as sf

# add root path as variable for easy imports
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
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
def patch_config(monkeypatch, sr, duration, request):
    from vocalization_classifier.src.prep_data import preprocess as pp
    import config

    node_path = str(getattr(request.node, "fspath", ""))

    # Default config
    n_fft = 128
    hop = 32
    n_mels = 16
    target_w = 32

    # Integration tests must have full size config
    if "/tests/integration/" in node_path:
        n_mels = 64
        target_w = 256
        n_fft = 512
        hop = 60

    # test smaller profile
    if "/tests/performance/" in node_path:
        n_fft = 256
        hop = 64
        n_mels = 32
        target_w = 64

    monkeypatch.setattr(pp, "SAMPLE_RATE", sr, raising=False)
    monkeypatch.setattr(pp, "DURATION_SEC", duration, raising=False)
    monkeypatch.setattr(pp, "N_FFT", n_fft, raising=False)
    monkeypatch.setattr(pp, "HOP_LENGTH", hop, raising=False)
    monkeypatch.setattr(pp, "N_MELS", n_mels, raising=False)
    monkeypatch.setattr(pp, "TARGET_W", target_w, raising=False)
    monkeypatch.setattr(pp, "SHOW_VISUALS", False, raising=False)

    audio_dir = getattr(config, "TEST_AUDIO_DIR", None)
    if audio_dir:
        audio_dir = str(audio_dir).strip()
        if audio_dir and os.path.isdir(audio_dir):
            monkeypatch.setattr(pp, "AUDIO_ROOT_PATH", audio_dir, raising=False)
    yield


# gives pytest the ability to write wav files to the mock audio root dir
@pytest.fixture
def write_wav(tmp_path):
    base = tmp_path / "audio_root"

    def _write(rel_path: str, data: np.ndarray, sr: int):
        out = tmp_path / "audio_root" / rel_path
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, data.astype(np.float32), sr, subtype="FLOAT")
        return str(out)

    _write.base = str(base)  # define a root for mock files to write to
    return _write
